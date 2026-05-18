"""
BRFSS Health Dashboard — single-file application.
Data loading, cleaning, feature engineering, static figures, and all
interactive callbacks are contained here. No external data files are
produced or required beyond the original CSV.

Run:  python3 dashboard.py
Open: http://127.0.0.1:8050
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

# ══════════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
CSV = ("Nutrition__Physical_Activity__and_Obesity_-_"
       "Behavioral_Risk_Factor_Surveillance_System.csv")

df_raw = pd.read_csv(CSV, low_memory=False)

# --- Step 1: fix trailing-space column name
df_raw.rename(columns={"High_Confidence_Limit ": "High_Confidence_Limit"},
              inplace=True)

# --- Step 2: short question labels
Q_LABELS = {
    "Percent of adults aged 18 years and older who have obesity":
        "Obesity",
    "Percent of adults aged 18 years and older who have an overweight classification":
        "Overweight",
    ("Percent of adults who achieve at least 150 minutes a week of moderate-intensity"
     " aerobic physical activity or 75 minutes a week of vigorous-intensity aerobic"
     " activity (or an equivalent combination)"):
        "150 min Aerobic (basic)",
    ("Percent of adults who achieve at least 150 minutes a week of moderate-intensity"
     " aerobic physical activity or 75 minutes a week of vigorous-intensity aerobic"
     " physical activity (or an equivalent combination) and engage in muscle-strengthening"
     " activities on 2 or more days a week"):
        "150 min Aerobic + Muscle",
    ("Percent of adults who achieve more than 300 minutes a week of moderate-intensity"
     " aerobic physical activity or 150 minutes a week of vigorous-intensity aerobic"
     " activity (or an equivalent combination)"):
        "300 min Aerobic (vigorous)",
    "Percent of adults who engage in muscle-strengthening activities on 2 or more days a week":
        "Muscle Strengthening",
    "Percent of adults who engage in no leisure-time physical activity":
        "No Leisure Activity",
    "Percent of adults who report consuming fruit less than one time daily":
        "Low Fruit Intake",
    "Percent of adults who report consuming vegetables less than one time daily":
        "Low Vegetable Intake",
}
df_raw["QuestionLabel"] = df_raw["Question"].map(Q_LABELS)

# --- Step 3: drop rows with missing Data_Value (CDC-suppressed small samples)
df = df_raw.dropna(subset=["Data_Value"]).copy()

# --- Step 4: cast types
df["YearStart"] = df["YearStart"].astype(int)
df["YearEnd"]   = df["YearEnd"].astype(int)

# --- Step 5: feature — CI width
df["CI_Width"] = df["High_Confidence_Limit"] - df["Low_Confidence_Limit"]

# --- Step 6: region mapping
REGION_MAP = {
    "AL":"South","AR":"South","DC":"South","DE":"South","FL":"South",
    "GA":"South","KY":"South","LA":"South","MD":"South","MS":"South",
    "NC":"South","OK":"South","SC":"South","TN":"South","TX":"South",
    "VA":"South","WV":"South",
    "CT":"Northeast","MA":"Northeast","ME":"Northeast","NH":"Northeast",
    "NJ":"Northeast","NY":"Northeast","PA":"Northeast","RI":"Northeast","VT":"Northeast",
    "IA":"Midwest","IL":"Midwest","IN":"Midwest","KS":"Midwest","MI":"Midwest",
    "MN":"Midwest","MO":"Midwest","ND":"Midwest","NE":"Midwest","OH":"Midwest",
    "SD":"Midwest","WI":"Midwest",
    "AK":"West","AZ":"West","CA":"West","CO":"West","HI":"West","ID":"West",
    "MT":"West","NM":"West","NV":"West","OR":"West","UT":"West","WA":"West","WY":"West",
}
df["Region"] = df["LocationAbbr"].map(REGION_MAP).fillna("Territory")

# Convenience lookups
YEARS      = [int(y) for y in sorted(df["YearStart"].unique())]
STATES     = sorted(df["LocationDesc"].unique())
Q_OPTIONS  = sorted(df["QuestionLabel"].dropna().unique())
STRAT_CATS = [c for c in sorted(df["StratificationCategory1"].unique()) if c != "Total"]

INCOME_ORDER = ["Less than $15,000","$15,000 - $24,999","$25,000 - $34,999",
                "$35,000 - $49,999","$50,000 - $74,999","$75,000 or greater"]
EDU_ORDER    = ["Less than high school","High school graduate",
                "Some college or technical school","College graduate"]
AGE_ORDER    = ["18 - 24","25 - 34","35 - 44","45 - 54","55 - 64","65 or older"]

# ══════════════════════════════════════════════════════════════════════════════
#  2. PRE-COMPUTE STATIC FIGURES  (Data Cleaning tab)
# ══════════════════════════════════════════════════════════════════════════════

# Missing-value bar chart (on raw data)
_miss_cols = ["Data_Value","Low_Confidence_Limit","High_Confidence_Limit",
              "Sample_Size","GeoLocation","Age(years)","Education","Sex",
              "Income","Race/Ethnicity","Data_Value_Unit","Data_Value_Footnote"]
_miss_pct  = [(c, round(df_raw[c].isnull().sum() / len(df_raw) * 100, 1))
              for c in _miss_cols]

fig_missing = go.Figure(go.Bar(
    x=[m[0] for m in _miss_pct],
    y=[m[1] for m in _miss_pct],
    marker_color=["#C0622B" if m[1] > 50 else "#2C5282" for m in _miss_pct],
    text=[f"{m[1]}%" for m in _miss_pct],
    textposition="outside",
))
fig_missing.update_layout(
    title="Missing Values by Column (%)",
    xaxis_title="Column", yaxis_title="Missing (%)",
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=12), yaxis=dict(range=[0, 115]),
    margin=dict(t=50, b=100),
)

# Summary stats table data
_stats = (df.groupby("QuestionLabel")["Data_Value"]
           .agg(Count="count",
                Mean=lambda x: round(x.mean(), 2),
                Std=lambda x: round(x.std(), 2),
                Min="min",
                Median="median",
                Q1=lambda x: round(x.quantile(0.25), 2),
                Q3=lambda x: round(x.quantile(0.75), 2),
                Max="max")
           .reset_index())

# Distribution box plot
_colors = px.colors.qualitative.Set2
fig_dist = go.Figure()
for i, (lbl, grp) in enumerate(df.groupby("QuestionLabel")):
    fig_dist.add_trace(go.Box(
        y=grp["Data_Value"], name=lbl,
        marker_color=_colors[i % len(_colors)],
        boxmean=True,
    ))
fig_dist.update_layout(
    title="Distribution of Values by Health Indicator",
    yaxis_title="% Adults", xaxis_tickangle=-20,
    plot_bgcolor="white", paper_bgcolor="white",
    showlegend=False, height=400, margin=dict(b=140),
)

# National trend (for cleaning tab context chart)
_nat = (df[df["StratificationCategory1"] == "Total"]
        .groupby(["YearStart", "QuestionLabel"])["Data_Value"]
        .mean().reset_index())
fig_nat_trend = px.line(
    _nat[_nat["QuestionLabel"].isin(["Obesity", "No Leisure Activity",
                                      "Low Fruit Intake", "Low Vegetable Intake"])],
    x="YearStart", y="Data_Value", color="QuestionLabel",
    title="National Trends 2011–2024 (Total stratification)",
    labels={"Data_Value": "% Adults", "YearStart": "Year",
            "QuestionLabel": "Indicator"},
    markers=True,
)
fig_nat_trend.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    hovermode="x unified", height=380,
    legend=dict(orientation="h", yanchor="top", y=-0.2),
)

# ══════════════════════════════════════════════════════════════════════════════
#  3. COLOUR / STYLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
PRIMARY   = "#C0622B"
SECONDARY = "#2C5282"
GREEN     = "#276749"
BG_LIGHT  = "#FFF8F3"
BG_CARD   = "#FFFFFF"
TEXT_DARK = "#1A202C"
GREY      = "#718096"

TAB_STYLE = {
    "fontWeight": "600", "color": GREY,
    "borderTop": "3px solid transparent",
    "backgroundColor": BG_LIGHT, "padding": "12px 20px",
}
TAB_ACTIVE = {
    **TAB_STYLE,
    "color": PRIMARY,
    "borderTop": f"3px solid {PRIMARY}",
    "backgroundColor": BG_CARD,
}

# ══════════════════════════════════════════════════════════════════════════════
#  4. LAYOUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def card(children, **style):
    return dbc.Card(
        dbc.CardBody(children),
        className="shadow-sm mb-3",
        style={"borderRadius": "10px", "backgroundColor": BG_CARD, **style},
    )

def section_title(text, sub=None):
    els = [html.H4(text, style={"color": PRIMARY, "fontWeight": "700",
                                 "marginBottom": "4px"})]
    if sub:
        els.append(html.P(sub, style={"color": GREY, "fontSize": "14px",
                                       "marginBottom": "16px"}))
    return html.Div(els)

def kpi(label, value, color=PRIMARY):
    return dbc.Col(
        dbc.Card([dbc.CardBody([
            html.H2(value, style={"color": color, "fontWeight": "800",
                                   "marginBottom": "4px"}),
            html.P(label, style={"color": GREY, "fontSize": "13px", "margin": 0}),
        ])], style={"borderRadius": "10px", "borderLeft": f"5px solid {color}",
                    "backgroundColor": BG_CARD,
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.07)"}),
        md=3, sm=6, className="mb-3",
    )

_tab_pad = {"backgroundColor": BG_LIGHT, "padding": "24px"}

# ══════════════════════════════════════════════════════════════════════════════
#  5. TAB CONTENTS  (all rendered up-front so callbacks always fire)
# ══════════════════════════════════════════════════════════════════════════════

# ── Tab 1: Background & Introduction ─────────────────────────────────────────
tab_intro = dbc.Container([
    dbc.Row([dbc.Col([
        html.H2("Nutrition, Physical Activity & Obesity",
                style={"color": PRIMARY, "fontWeight": "800", "fontSize": "28px"}),
        html.H5("Behavioral Risk Factor Surveillance System (BRFSS) — CDC",
                style={"color": SECONDARY, "marginBottom": "16px"}),
        html.P("The BRFSS is the nation's premier system of health-related telephone "
               "surveys that collect state data about U.S. residents regarding their "
               "health-related risk behaviors, chronic health conditions, and use of "
               "preventive services. This dataset covers adults aged 18+ across all "
               "50 states, Washington D.C., and U.S. territories from 2011 to 2024.",
               style={"color": TEXT_DARK, "lineHeight": "1.7", "fontSize": "15px"}),
        html.P("The data focuses on three major public health concern areas: "
               "Obesity/Weight Status, Physical Activity, and Fruits & Vegetables "
               "consumption. Each observation is stratified by demographic factors "
               "including age, sex, income, education, and race/ethnicity.",
               style={"color": TEXT_DARK, "lineHeight": "1.7", "fontSize": "15px"}),
    ], md=12)], className="mb-3"),

    dbc.Row([
        kpi("Total Records",  "110,880",    PRIMARY),
        kpi("Years Covered",  "2011–2024",  SECONDARY),
        kpi("U.S. Locations", "55",         GREEN),
        kpi("Health Topics",  "9 Questions","#744210"),
    ]),

    dbc.Row([dbc.Col(card([
        section_title("Dataset Column Reference"),
        dash_table.DataTable(
            data=[
                {"Column": "YearStart / YearEnd",             "Description": "Survey year"},
                {"Column": "LocationAbbr / LocationDesc",     "Description": "State abbreviation and full name"},
                {"Column": "Class",                           "Description": "Obesity · Physical Activity · Fruits & Vegetables"},
                {"Column": "Question",                        "Description": "Specific health behavior question (9 total)"},
                {"Column": "Data_Value",                      "Description": "Percentage (%) of adults meeting the criterion"},
                {"Column": "Low / High_Confidence_Limit",     "Description": "95% confidence interval bounds"},
                {"Column": "Sample_Size",                     "Description": "Survey respondents for this record"},
                {"Column": "StratificationCategory1",         "Description": "Demographic dimension (Age, Sex, Income, Education, Race/Ethnicity, Total)"},
                {"Column": "Stratification1",                 "Description": "Specific group within the dimension"},
            ],
            columns=[{"name": c, "id": c} for c in ["Column", "Description"]],
            style_cell={"textAlign": "left", "padding": "8px 12px", "fontSize": "13px"},
            style_header={"backgroundColor": PRIMARY, "color": "white", "fontWeight": "700"},
            style_data_conditional=[{"if": {"row_index": "odd"},
                                     "backgroundColor": "#FFF3EC"}],
            style_table={"overflowX": "auto"},
        ),
    ]))]),

    dbc.Row([dbc.Col(card([
        section_title("Three Health Topic Classes"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H5("⚖ Obesity / Weight Status", style={"color": PRIMARY}),
                html.P("Tracks adults classified as obese (BMI ≥ 30) or overweight (BMI 25–29.9). "
                       "Obesity rates have risen from 27.6% in 2011 to 34.0% in 2024.",
                       style={"fontSize": "13px", "color": TEXT_DARK}),
            ])], style={"border": f"2px solid {PRIMARY}", "borderRadius": "10px"}), md=4),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H5("🏃 Physical Activity", style={"color": SECONDARY}),
                html.P("Measures aerobic exercise adherence, muscle-strengthening, and "
                       "no leisure-time physical activity (~25.8% of all adults).",
                       style={"fontSize": "13px", "color": TEXT_DARK}),
            ])], style={"border": f"2px solid {SECONDARY}", "borderRadius": "10px"}), md=4),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H5("🥦 Fruits & Vegetables", style={"color": GREEN}),
                html.P("Captures daily consumption habits. ~39.8% eat fruit less than once "
                       "daily; ~21.2% eat vegetables less than once per day.",
                       style={"fontSize": "13px", "color": TEXT_DARK}),
            ])], style={"border": f"2px solid {GREEN}", "borderRadius": "10px"}), md=4),
        ])
    ]))]),
], fluid=True, style=_tab_pad)


# ── Tab 2: Key Questions ──────────────────────────────────────────────────────
def q_card(num, q_type, color, questions, icon):
    return dbc.Col(dbc.Card([
        dbc.CardHeader(
            html.Div([
                html.Span(icon, style={"fontSize": "28px", "marginRight": "10px"}),
                html.Div([
                    html.Small(f"TYPE {num}",
                               style={"color": "rgba(255,255,255,0.8)",
                                      "fontSize": "11px", "fontWeight": "700",
                                      "letterSpacing": "1px"}),
                    html.H5(q_type, style={"color": "white", "margin": 0,
                                            "fontWeight": "700"}),
                ])
            ], style={"display": "flex", "alignItems": "center"}),
            style={"backgroundColor": color, "borderRadius": "10px 10px 0 0",
                   "padding": "16px"}
        ),
        dbc.CardBody([
            html.Ul([
                html.Li(q, style={"marginBottom": "10px", "color": TEXT_DARK,
                                   "fontSize": "14px", "lineHeight": "1.5"})
                for q in questions
            ], style={"paddingLeft": "18px"})
        ], style={"backgroundColor": BG_CARD}),
    ], style={"border": f"2px solid {color}", "borderRadius": "10px",
              "boxShadow": "0 4px 12px rgba(0,0,0,0.08)"}), md=4, className="mb-3")

tab_questions = dbc.Container([
    dbc.Row([dbc.Col([
        html.H3("Key Questions to Explore",
                style={"color": PRIMARY, "fontWeight": "800"}),
        html.P("Three categories of questions drive the analysis — each answered by "
               "a dedicated dashboard tab.",
               style={"color": GREY, "fontSize": "15px", "marginBottom": "24px"}),
    ])]),
    dbc.Row([
        q_card(1, "Temporal / Trend Questions", PRIMARY,
               ["Q1: How have national obesity rates changed from 2011 to 2024?",
                "Q2: Has physical inactivity improved or worsened over the study period?",
                "Q3: Are American adults eating more or fewer fruits and vegetables over time?"],
               "📈"),
        q_card(2, "Geographic / Spatial Questions", SECONDARY,
               ["Q4: Which states have the highest and lowest obesity rates?",
                "Q5: Which states show the highest levels of physical inactivity?",
                "Q6: How do fruit and vegetable habits vary geographically?"],
               "🗺"),
        q_card(3, "Sociodemographic Questions", GREEN,
               ["Q7: How do income and education levels influence obesity?",
                "Q8: Do racial and ethnic disparities exist in these outcomes?",
                "Q9: How do health behaviors differ by age group and sex?"],
               "👥"),
    ]),
    dbc.Row([dbc.Col(card([
        section_title("Why These Questions Matter"),
        dbc.Row([
            dbc.Col([
                html.H6("Public Health Impact", style={"color": PRIMARY, "fontWeight": "700"}),
                html.P("Obesity is linked to heart disease, type 2 diabetes, stroke, and "
                       "cancers — understanding trends guides resource allocation.",
                       style={"fontSize": "13px", "color": TEXT_DARK}),
            ], md=4),
            dbc.Col([
                html.H6("Policy Relevance", style={"color": SECONDARY, "fontWeight": "700"}),
                html.P("Geographic patterns reveal where interventions are most needed — "
                       "from community exercise programs to food access initiatives.",
                       style={"fontSize": "13px", "color": TEXT_DARK}),
            ], md=4),
            dbc.Col([
                html.H6("Health Equity", style={"color": GREEN, "fontWeight": "700"}),
                html.P("Disparities across race, income, and education highlight systemic "
                       "inequities in healthcare access and healthy lifestyle opportunities.",
                       style={"fontSize": "13px", "color": TEXT_DARK}),
            ], md=4),
        ])
    ]))]),
], fluid=True, style=_tab_pad)


# ── Tab 3: Data Cleaning & Stats ──────────────────────────────────────────────
tab_cleaning = dbc.Container([
    dbc.Row([dbc.Col([
        html.H3("Data Cleaning, Statistics & Analysis",
                style={"color": PRIMARY, "fontWeight": "800"}),
        html.P("All processing happens inside dashboard.py — no separate scripts needed.",
               style={"color": GREY, "fontSize": "15px", "marginBottom": "20px"}),
    ])]),

    dbc.Row([
        dbc.Col(card([
            section_title("Cleaning Steps Applied"),
            html.Ol([
                html.Li([html.Strong("Column name fix: "),
                         "Removed trailing space from 'High_Confidence_Limit '"],
                        style={"marginBottom": "8px", "fontSize": "14px"}),
                html.Li([html.Strong("Dropped missing Data_Value (11.9%): "),
                         "CDC suppresses values when sample size < 50 or data is unreliable."],
                        style={"marginBottom": "8px", "fontSize": "14px"}),
                html.Li([html.Strong("Type casting: "),
                         "YearStart/YearEnd → int; Sample_Size → numeric."],
                        style={"marginBottom": "8px", "fontSize": "14px"}),
                html.Li([html.Strong("Sparse demographic columns: "),
                         "Age, Education, Sex, Income, Race/Ethnicity are sparse by design — "
                         "each row uses one stratification stored in StratificationCategory1."],
                        style={"marginBottom": "8px", "fontSize": "14px"}),
                html.Li([html.Strong("Added QuestionLabel: "),
                         "Mapped long question strings to short readable labels."],
                        style={"marginBottom": "8px", "fontSize": "14px"}),
                html.Li([html.Strong("Added CI_Width: "),
                         "High_Confidence_Limit − Low_Confidence_Limit (precision indicator)."],
                        style={"marginBottom": "8px", "fontSize": "14px"}),
                html.Li([html.Strong("Added Region: "),
                         "U.S. Census regions (South / Northeast / Midwest / West / Territory)."],
                        style={"marginBottom": "8px", "fontSize": "14px"}),
                html.Li([html.Strong("Zero duplicates: "),
                         "Confirmed with exact and natural-key duplicate checks."],
                        style={"marginBottom": "8px", "fontSize": "14px"}),
            ])
        ]), md=7),

        dbc.Col(card([
            section_title("Post-Cleaning Summary"),
            html.Div([html.H3("97,666",   style={"color": PRIMARY,   "fontWeight": "800", "marginBottom": "2px"}),
                      html.Small("Clean records (−11.9%)", style={"color": GREY})], className="mb-3"),
            html.Div([html.H3("0",        style={"color": SECONDARY, "fontWeight": "800", "marginBottom": "2px"}),
                      html.Small("Duplicate rows", style={"color": GREY})], className="mb-3"),
            html.Div([html.H3("0",        style={"color": GREEN,     "fontWeight": "800", "marginBottom": "2px"}),
                      html.Small("Data_Value outside [0, 100]", style={"color": GREY})], className="mb-3"),
            html.Div([html.H3("+3 cols",  style={"color": "#744210", "fontWeight": "800", "marginBottom": "2px"}),
                      html.Small("New features: QuestionLabel, CI_Width, Region",
                                 style={"color": GREY})]),
        ]), md=5),
    ]),

    dbc.Row([dbc.Col(card([dcc.Graph(figure=fig_missing, id="fig-missing")]))]),

    dbc.Row([dbc.Col(card([
        section_title("Summary Statistics by Indicator"),
        dash_table.DataTable(
            id="stats-table",
            data=_stats.to_dict("records"),
            columns=[{"name": c, "id": c} for c in _stats.columns],
            style_cell={"textAlign": "center", "padding": "8px 10px", "fontSize": "12px"},
            style_cell_conditional=[{"if": {"column_id": "QuestionLabel"},
                                     "textAlign": "left", "minWidth": "200px"}],
            style_header={"backgroundColor": PRIMARY, "color": "white", "fontWeight": "700"},
            style_data_conditional=[{"if": {"row_index": "odd"},
                                     "backgroundColor": "#FFF3EC"}],
            style_table={"overflowX": "auto"},
        ),
    ]))]),

    dbc.Row([dbc.Col(card([dcc.Graph(figure=fig_dist, id="fig-dist")]))]),
    dbc.Row([dbc.Col(card([dcc.Graph(figure=fig_nat_trend, id="fig-nat-trend")]))]),
], fluid=True, style=_tab_pad)


# ── Tab 4: Temporal Trends ────────────────────────────────────────────────────
tab_temporal = dbc.Container([
    dbc.Row([dbc.Col([
        html.H3("Temporal Trends", style={"color": PRIMARY, "fontWeight": "800"}),
        html.P("How have health behaviors changed 2011–2024? (Addresses Q1, Q2, Q3)",
               style={"color": GREY, "fontSize": "15px", "marginBottom": "20px"}),
    ])]),
    dbc.Row([
        dbc.Col(card([
            html.Label("Health Indicator", style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(id="trend-question",
                         options=[{"label": v, "value": v} for v in Q_OPTIONS],
                         value="Obesity", clearable=False),
        ]), md=4),
        dbc.Col(card([
            html.Label("Stratification Category",
                       style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(id="trend-strat-cat",
                         options=[{"label": "Total (National)", "value": "Total"}] +
                                 [{"label": c, "value": c} for c in STRAT_CATS],
                         value="Total", clearable=False),
        ]), md=4),
        dbc.Col(card([
            html.Label("Highlight States (optional)",
                       style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(id="trend-states",
                         options=[{"label": s, "value": s} for s in STATES],
                         multi=True,
                         placeholder="Leave blank for national average"),
        ]), md=4),
    ]),
    dbc.Row([dbc.Col(card([dcc.Graph(id="trend-line-chart")]))]),
    dbc.Row([
        dbc.Col(card([dcc.Graph(id="trend-heatmap")]),    md=7),
        dbc.Col(card([dcc.Graph(id="trend-change-bar")]), md=5),
    ]),
], fluid=True, style=_tab_pad)


# ── Tab 5: Geographic Analysis ────────────────────────────────────────────────
tab_geographic = dbc.Container([
    dbc.Row([dbc.Col([
        html.H3("Geographic Analysis", style={"color": PRIMARY, "fontWeight": "800"}),
        html.P("State-level variation in health behaviors. (Addresses Q4, Q5, Q6)",
               style={"color": GREY, "fontSize": "15px", "marginBottom": "20px"}),
    ])]),
    dbc.Row([
        dbc.Col(card([
            html.Label("Health Indicator",
                       style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(id="geo-question",
                         options=[{"label": v, "value": v} for v in Q_OPTIONS],
                         value="Obesity", clearable=False),
        ]), md=4),
        dbc.Col(card([
            html.Label("Year", style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Slider(id="geo-year",
                       min=min(YEARS), max=max(YEARS), step=1,
                       value=max(YEARS),
                       marks={y: str(y) for y in YEARS[::2]},
                       tooltip={"placement": "bottom"}),
        ]), md=8),
    ]),
    dbc.Row([dbc.Col(card([dcc.Graph(id="geo-map")]))]),
    dbc.Row([
        dbc.Col(card([dcc.Graph(id="geo-bar-top")]),    md=6),
        dbc.Col(card([dcc.Graph(id="geo-bar-bottom")]), md=6),
    ]),
    dbc.Row([dbc.Col(card([dcc.Graph(id="geo-scatter")]))]),
], fluid=True, style=_tab_pad)


# ── Tab 6: Demographic Analysis ───────────────────────────────────────────────
tab_demographic = dbc.Container([
    dbc.Row([dbc.Col([
        html.H3("Sociodemographic Analysis",
                style={"color": PRIMARY, "fontWeight": "800"}),
        html.P("How income, education, race/ethnicity, age, and sex shape outcomes. "
               "(Addresses Q7, Q8, Q9)",
               style={"color": GREY, "fontSize": "15px", "marginBottom": "20px"}),
    ])]),
    dbc.Row([
        dbc.Col(card([
            html.Label("Health Indicator",
                       style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(id="demo-question",
                         options=[{"label": v, "value": v} for v in Q_OPTIONS],
                         value="Obesity", clearable=False),
        ]), md=4),
        dbc.Col(card([
            html.Label("Year Range", style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.RangeSlider(id="demo-year-range",
                            min=min(YEARS), max=max(YEARS), step=1,
                            value=[min(YEARS), max(YEARS)],
                            marks={y: str(y) for y in YEARS[::3]},
                            tooltip={"placement": "bottom"}),
        ]), md=8),
    ]),
    dbc.Row([
        dbc.Col(card([dcc.Graph(id="demo-income")]),    md=6),
        dbc.Col(card([dcc.Graph(id="demo-education")]), md=6),
    ]),
    dbc.Row([
        dbc.Col(card([dcc.Graph(id="demo-race")]), md=6),
        dbc.Col(card([dcc.Graph(id="demo-age")]),  md=6),
    ]),
    dbc.Row([dbc.Col(card([dcc.Graph(id="demo-sex-trend")]))]),
], fluid=True, style=_tab_pad)


# ── Tab 7: Summary & Conclusions ─────────────────────────────────────────────
def answer_card(qnum, question, answer, color=PRIMARY):
    return dbc.Card([
        dbc.CardHeader(
            html.Div([
                html.Span(f"Q{qnum}", style={
                    "backgroundColor": color, "color": "white",
                    "borderRadius": "50%", "width": "32px", "height": "32px",
                    "display": "inline-flex", "alignItems": "center",
                    "justifyContent": "center", "fontWeight": "800",
                    "marginRight": "10px", "flexShrink": "0",
                }),
                html.Strong(question, style={"fontSize": "14px"}),
            ], style={"display": "flex", "alignItems": "center"}),
            style={"backgroundColor": f"{color}15",
                   "borderBottom": f"2px solid {color}"}
        ),
        dbc.CardBody(html.P(answer, style={"fontSize": "13px", "color": TEXT_DARK,
                                            "lineHeight": "1.6", "margin": 0})),
    ], className="mb-2",
       style={"borderRadius": "8px", "border": f"1px solid {color}40"})

tab_summary = dbc.Container([
    dbc.Row([dbc.Col([
        html.H3("Summary, Conclusions & Answered Questions",
                style={"color": PRIMARY, "fontWeight": "800"}),
        html.P("Key findings and direct answers to all 9 research questions.",
               style={"color": GREY, "fontSize": "15px", "marginBottom": "20px"}),
    ])]),

    dbc.Row([dbc.Col(card([
        section_title("Key Findings", "What the data reveals"),
        html.Ul([
            html.Li("Obesity has risen from 27.6% (2011) to 34.0% (2024) — a 23% relative increase.",
                    style={"marginBottom": "8px", "fontSize": "14px", "color": TEXT_DARK}),
            html.Li("Physical inactivity remains at ~25–26% with no national improvement 2011–2024.",
                    style={"marginBottom": "8px", "fontSize": "14px", "color": TEXT_DARK}),
            html.Li("Southern states (WV 41.2%, MS 40.1%) have the highest obesity rates; "
                    "Colorado (24.9%) and D.C. (23.5%) the lowest.",
                    style={"marginBottom": "8px", "fontSize": "14px", "color": TEXT_DARK}),
            html.Li("Strong income gradient: obesity 35.8% (<$15K) → 29.0% (≥$75K).",
                    style={"marginBottom": "8px", "fontSize": "14px", "color": TEXT_DARK}),
            html.Li("Education mirrors income: college graduates 25.4% vs. no-diploma 34.7%.",
                    style={"marginBottom": "8px", "fontSize": "14px", "color": TEXT_DARK}),
            html.Li("Largest racial disparities: Asian 11.6% vs. Hawaiian/Pacific Islander 40.8%.",
                    style={"marginBottom": "8px", "fontSize": "14px", "color": TEXT_DARK}),
            html.Li("Physical inactivity doubles from 17% (18–24 yrs) to 32% (65+ yrs).",
                    style={"marginBottom": "8px", "fontSize": "14px", "color": TEXT_DARK}),
        ])
    ]))]),

    dbc.Row([
        dbc.Col([
            html.H5("Type 1 — Trend Questions",
                    style={"color": PRIMARY, "fontWeight": "700", "marginBottom": "10px"}),
            answer_card(1, "How have obesity rates changed 2011–2024?",
                "Obesity rose 27.6% → 34.0% (+6.4 pp). Linear trend +0.47 pp/year (R²=0.97). "
                "Steepest rise 2019–2021 coincides with COVID-19 pandemic.", PRIMARY),
            answer_card(2, "Has physical inactivity improved?",
                "No. ~25–26% throughout the period with no meaningful national decline. "
                "Some western states show mild improvement post-2017.", PRIMARY),
            answer_card(3, "Are nutrition habits improving?",
                "No. Fruit <1×/day ≈ 40%; vegetable <1×/day ≈ 21%. Both flat across 14 years.",
                PRIMARY),
        ], md=4),
        dbc.Col([
            html.H5("Type 2 — Geographic Questions",
                    style={"color": SECONDARY, "fontWeight": "700", "marginBottom": "10px"}),
            answer_card(4, "Which states have highest / lowest obesity?",
                "Highest: WV 41.2%, MS 40.1%, AR 40.0%, LA 39.9%, AL 39.2%. "
                "Lowest: D.C. 23.5%, CO 24.9%, HI 26.1%. Clear South → West gradient.",
                SECONDARY),
            answer_card(5, "Which states have most physical inactivity?",
                "Southern states (MS, LA, TN, AL) lead in inactivity. "
                "Western states (CO, UT, OR) have the most active populations.", SECONDARY),
            answer_card(6, "How does nutrition vary geographically?",
                "Low fruit/veg intake worst in South & Midwest. "
                "New England and Pacific Coast states report better habits.", SECONDARY),
        ], md=4),
        dbc.Col([
            html.H5("Type 3 — Sociodemographic Questions",
                    style={"color": GREEN, "fontWeight": "700", "marginBottom": "10px"}),
            answer_card(7, "How do income & education affect obesity?",
                "Strong inverse for both. <$15K → 35.8%; ≥$75K → 29.0%. "
                "College graduates 25.4% vs. no diploma 34.7%.", GREEN),
            answer_card(8, "Do racial/ethnic disparities exist?",
                "Yes. Hawaiian/Pacific Islander 40.8%, Non-Hispanic Black 38.7% — "
                "highest burden. Asian adults lowest at 11.6%.", GREEN),
            answer_card(9, "How do age and sex differ?",
                "Inactivity doubles 17% (18–24) → 32% (65+). "
                "Male (30.8%) and female (31.1%) obesity nearly equal nationally.", GREEN),
        ], md=4),
    ]),

    dbc.Row([dbc.Col(card([
        section_title("Policy Recommendations"),
        dbc.Row([
            dbc.Col([html.H6("1. Target Southern States", style={"color": PRIMARY, "fontWeight": "700"}),
                     html.P("Dedicate federal funding for obesity prevention, healthy food access, "
                            "and community exercise infrastructure in WV, MS, AR, LA.",
                            style={"fontSize": "13px", "color": TEXT_DARK})], md=3),
            dbc.Col([html.H6("2. Address Income Inequality", style={"color": SECONDARY, "fontWeight": "700"}),
                     html.P("Expand SNAP, support community gardens in low-income neighborhoods, "
                            "and subsidize fitness access for lower-income adults.",
                            style={"fontSize": "13px", "color": TEXT_DARK})], md=3),
            dbc.Col([html.H6("3. Culturally Tailored Programs", style={"color": GREEN, "fontWeight": "700"}),
                     html.P("Develop culturally appropriate interventions for Black, Hispanic, "
                            "and Native American communities who bear disproportionate burden.",
                            style={"fontSize": "13px", "color": TEXT_DARK})], md=3),
            dbc.Col([html.H6("4. Senior Activity Programs", style={"color": "#744210", "fontWeight": "700"}),
                     html.P("Invest in senior-friendly exercise and nutrition programs — "
                            "inactivity doubles from young adults to age 65+.",
                            style={"fontSize": "13px", "color": TEXT_DARK})], md=3),
        ])
    ]))]),
], fluid=True, style=_tab_pad)


# ══════════════════════════════════════════════════════════════════════════════
#  6. APP LAYOUT — all tab contents in DOM from the start
# ══════════════════════════════════════════════════════════════════════════════
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="BRFSS Health Dashboard",
)
server = app.server   # expose for production deployment if needed

app.layout = html.Div([
    # Header
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Nutrition, Physical Activity & Obesity",
                            style={"color": "white", "fontWeight": "800",
                                   "fontSize": "22px", "margin": 0}),
                    html.Small("CDC BRFSS | 2011–2024 | DSA 5200",
                               style={"color": "rgba(255,255,255,0.8)", "fontSize": "12px"}),
                ], md=9),
                dbc.Col(html.Div("ADV Project Dashboard",
                                  style={"color": "rgba(255,255,255,0.7)", "fontSize": "12px",
                                         "textAlign": "right", "marginTop": "6px"}), md=3),
            ], align="center"),
        ], fluid=True),
    ], style={"backgroundColor": PRIMARY, "padding": "14px 0",
              "boxShadow": "0 2px 8px rgba(0,0,0,0.2)"}),

    # Tabs — children embedded directly so all components are always in DOM
    dcc.Tabs(id="main-tabs", value="intro", children=[
        dcc.Tab(label="Background & Introduction", value="intro",
                children=[tab_intro],
                style=TAB_STYLE, selected_style=TAB_ACTIVE),
        dcc.Tab(label="Key Questions", value="questions",
                children=[tab_questions],
                style=TAB_STYLE, selected_style=TAB_ACTIVE),
        dcc.Tab(label="Data Cleaning & Stats", value="cleaning",
                children=[tab_cleaning],
                style=TAB_STYLE, selected_style=TAB_ACTIVE),
        dcc.Tab(label="Trend Analysis", value="temporal",
                children=[tab_temporal],
                style=TAB_STYLE, selected_style=TAB_ACTIVE),
        dcc.Tab(label="Geographic Analysis", value="geographic",
                children=[tab_geographic],
                style=TAB_STYLE, selected_style=TAB_ACTIVE),
        dcc.Tab(label="Demographic Analysis", value="demographic",
                children=[tab_demographic],
                style=TAB_STYLE, selected_style=TAB_ACTIVE),
        dcc.Tab(label="Summary & Conclusions", value="summary",
                children=[tab_summary],
                style=TAB_STYLE, selected_style=TAB_ACTIVE),
    ], style={"backgroundColor": BG_LIGHT, "borderBottom": "2px solid #E2E8F0"}),

], style={"fontFamily": "'Segoe UI', system-ui, sans-serif",
          "backgroundColor": BG_LIGHT})


# ══════════════════════════════════════════════════════════════════════════════
#  7. CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

# ── Temporal Trends ───────────────────────────────────────────────────────────
@app.callback(
    Output("trend-line-chart",  "figure"),
    Output("trend-heatmap",     "figure"),
    Output("trend-change-bar",  "figure"),
    Input("trend-question",   "value"),
    Input("trend-strat-cat",  "value"),
    Input("trend-states",     "value"),
)
def cb_trends(question, strat_cat, states):
    sub = df[(df["QuestionLabel"] == question) &
             (df["StratificationCategory1"] == strat_cat)]

    # Line chart
    if states:
        grp = (sub[sub["LocationDesc"].isin(states)]
               .groupby(["YearStart", "LocationDesc"])["Data_Value"]
               .mean().reset_index())
        fig_line = px.line(grp, x="YearStart", y="Data_Value", color="LocationDesc",
                           title=f"{question} — Selected States",
                           labels={"Data_Value": "% Adults", "YearStart": "Year"},
                           markers=True)
    else:
        nat = sub.groupby("YearStart")["Data_Value"].mean().reset_index()
        fig_line = px.line(nat, x="YearStart", y="Data_Value",
                           title=f"{question} — National Average 2011–2024",
                           labels={"Data_Value": "% Adults", "YearStart": "Year"},
                           markers=True)
        fig_line.update_traces(line=dict(color=PRIMARY, width=3),
                                marker=dict(size=8, color=PRIMARY))
        if strat_cat == "Total":
            ci = df_raw[(df_raw["QuestionLabel"] == question) &
                        (df_raw["StratificationCategory1"] == strat_cat)]
            lo = ci.groupby("YearStart")["Low_Confidence_Limit"].mean()
            hi = ci.groupby("YearStart")["High_Confidence_Limit"].mean()
            yrs = lo.index.tolist()
            fig_line.add_trace(go.Scatter(
                x=yrs + yrs[::-1],
                y=list(hi.values) + list(lo.values[::-1]),
                fill="toself", fillcolor="rgba(192,98,43,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI", hoverinfo="skip",
            ))

    fig_line.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=12), hovermode="x unified", height=380,
        legend=dict(orientation="h", yanchor="top", y=-0.2),
    )
    fig_line.update_xaxes(showgrid=True, gridcolor="#F0F0F0")
    fig_line.update_yaxes(showgrid=True, gridcolor="#F0F0F0")

    # Heatmap (always uses Total for the state grid)
    ht = (df[(df["QuestionLabel"] == question) &
             (df["StratificationCategory1"] == "Total")]
          .groupby(["LocationAbbr", "YearStart"])["Data_Value"]
          .mean().unstack())
    ht = ht.loc[ht.mean(axis=1).sort_values(ascending=False).head(25).index]
    fig_heat = go.Figure(go.Heatmap(
        z=ht.values,
        x=[str(c) for c in ht.columns],
        y=ht.index.tolist(),
        colorscale="RdYlGn_r",
        text=np.where(np.isnan(ht.values), "", np.round(ht.values, 1)),
        texttemplate="%{text}",
        hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
    ))
    fig_heat.update_layout(
        title=f"State × Year Heatmap — {question} (Top 25 States)",
        xaxis_title="Year", yaxis_title="State",
        plot_bgcolor="white", paper_bgcolor="white",
        height=520, margin=dict(l=60, r=20, t=50, b=40),
    )

    # Change bar (first → last year)
    chg = df[(df["QuestionLabel"] == question) &
             (df["StratificationCategory1"] == "Total")]
    yr_min = chg["YearStart"].min()
    yr_max = chg["YearStart"].max()
    first = (chg[chg["YearStart"] == yr_min]
             .groupby("LocationAbbr")["Data_Value"].mean())
    last  = (chg[chg["YearStart"] == yr_max]
             .groupby("LocationAbbr")["Data_Value"].mean())
    delta = (last - first).dropna().sort_values()
    fig_chg = go.Figure(go.Bar(
        x=delta.values, y=delta.index, orientation="h",
        marker_color=[PRIMARY if v >= 0 else SECONDARY for v in delta.values],
        text=[f"{v:+.1f}%" for v in delta.values],
        textposition="outside",
    ))
    fig_chg.update_layout(
        title=f"{question}: Change {yr_min}→{yr_max}",
        xaxis_title="pp Change",
        plot_bgcolor="white", paper_bgcolor="white",
        height=520, margin=dict(l=60, r=70, t=50, b=40),
        font=dict(size=10),
    )
    return fig_line, fig_heat, fig_chg


# ── Geographic Analysis ───────────────────────────────────────────────────────
@app.callback(
    Output("geo-map",        "figure"),
    Output("geo-bar-top",    "figure"),
    Output("geo-bar-bottom", "figure"),
    Output("geo-scatter",    "figure"),
    Input("geo-question", "value"),
    Input("geo-year",     "value"),
)
def cb_geo(question, year):
    sub = df[(df["QuestionLabel"] == question) &
             (df["YearStart"] == year) &
             (df["StratificationCategory1"] == "Total")]
    sv  = sub.groupby(["LocationAbbr", "LocationDesc"])["Data_Value"].mean().reset_index()

    # Choropleth
    fig_map = px.choropleth(
        sv, locations="LocationAbbr", locationmode="USA-states",
        color="Data_Value", scope="usa", color_continuous_scale="RdYlGn_r",
        hover_name="LocationDesc",
        hover_data={"Data_Value": ":.1f", "LocationAbbr": False},
        labels={"Data_Value": "% Adults"},
        title=f"{question} by State — {year}",
    )
    fig_map.update_layout(
        coloraxis_colorbar=dict(title="% Adults"),
        paper_bgcolor="white", height=440,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # Top / Bottom 10 bars
    sv_clean = sv.dropna(subset=["Data_Value"]).sort_values("Data_Value", ascending=False)
    top10    = sv_clean.head(10)
    bot10    = sv_clean.tail(10).sort_values("Data_Value")

    def make_bar(data, title, scale):
        fig = px.bar(data, x="Data_Value", y="LocationDesc", orientation="h",
                     title=title,
                     labels={"Data_Value": "% Adults", "LocationDesc": ""},
                     color="Data_Value", color_continuous_scale=scale,
                     text=data["Data_Value"].round(1))
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=10, r=70, t=50, b=20), height=380)
        return fig

    fig_top = make_bar(top10, f"Top 10 States — {question} ({year})",    "Reds")
    fig_bot = make_bar(bot10, f"Bottom 10 States — {question} ({year})", "Greens_r")

    # Scatter: obesity vs inactivity
    ob  = (df[(df["QuestionLabel"] == "Obesity") &
              (df["YearStart"] == year) &
              (df["StratificationCategory1"] == "Total")]
           .groupby("LocationDesc")["Data_Value"].mean().rename("Obesity"))
    act = (df[(df["QuestionLabel"] == "No Leisure Activity") &
              (df["YearStart"] == year) &
              (df["StratificationCategory1"] == "Total")]
           .groupby("LocationDesc")["Data_Value"].mean().rename("No Leisure Activity"))
    scat = pd.concat([ob, act], axis=1).dropna().reset_index()

    fig_scat = px.scatter(
        scat, x="No Leisure Activity", y="Obesity",
        text="LocationDesc",
        title=f"Obesity vs. Physical Inactivity by State ({year})",
        labels={"No Leisure Activity": "No Leisure Activity (%)",
                "Obesity": "Obesity Rate (%)"},
        trendline="ols",
        color="Obesity", color_continuous_scale="RdYlGn_r",
    )
    fig_scat.update_traces(textposition="top center", textfont_size=9)
    fig_scat.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                            height=480, coloraxis_showscale=False)
    return fig_map, fig_top, fig_bot, fig_scat


# ── Demographic Analysis ──────────────────────────────────────────────────────
@app.callback(
    Output("demo-income",    "figure"),
    Output("demo-education", "figure"),
    Output("demo-race",      "figure"),
    Output("demo-age",       "figure"),
    Output("demo-sex-trend", "figure"),
    Input("demo-question",   "value"),
    Input("demo-year-range", "value"),
)
def cb_demo(question, yr_range):
    sub = df[(df["QuestionLabel"] == question) &
             (df["YearStart"] >= yr_range[0]) &
             (df["YearStart"] <= yr_range[1])]

    def ordered_bar(data, strat_cat, order_list, title, x_col="Stratification1",
                    color_scale="RdYlGn_r", vertical=True):
        grp = data[data["StratificationCategory1"] == strat_cat]
        if grp.empty:
            return go.Figure()
        m = grp.groupby(x_col)["Data_Value"].mean().reset_index()
        m["_ord"] = m[x_col].apply(lambda v: order_list.index(v)
                                   if v in order_list else 99)
        m = m.sort_values("_ord")
        if vertical:
            fig = px.bar(m, x=x_col, y="Data_Value",
                         title=title,
                         labels={"Data_Value": "% Adults", x_col: ""},
                         color="Data_Value", color_continuous_scale=color_scale,
                         text=m["Data_Value"].round(1))
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(xaxis_tickangle=-30,
                              margin=dict(b=80, t=50), height=380)
        else:
            m = m.sort_values("Data_Value")
            fig = px.bar(m, x="Data_Value", y=x_col, orientation="h",
                         title=title,
                         labels={"Data_Value": "% Adults", x_col: ""},
                         color="Data_Value", color_continuous_scale=color_scale,
                         text=m["Data_Value"].round(1))
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(margin=dict(l=10, r=70, t=50, b=20), height=380)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          coloraxis_showscale=False, showlegend=False)
        return fig

    fig_inc = ordered_bar(sub, "Income",      INCOME_ORDER,
                          f"{question} by Income Level",    vertical=True)
    fig_edu = ordered_bar(sub, "Education",   EDU_ORDER,
                          f"{question} by Education Level", vertical=True)
    fig_race = ordered_bar(sub, "Race/Ethnicity", [],
                           f"{question} by Race/Ethnicity", vertical=False)
    fig_age  = ordered_bar(sub, "Age (years)", AGE_ORDER,
                           f"{question} by Age Group",      vertical=True)

    # Sex trend over full dataset (ignores year range for full picture)
    sx = (df[(df["QuestionLabel"] == question) &
             (df["StratificationCategory1"] == "Sex") &
             (df["Stratification1"].isin(["Male", "Female"]))]
          .groupby(["YearStart", "Stratification1"])["Data_Value"]
          .mean().reset_index())
    fig_sex = px.line(sx, x="YearStart", y="Data_Value", color="Stratification1",
                      title=f"{question} Over Time by Sex (all years)",
                      labels={"Data_Value": "% Adults", "YearStart": "Year",
                               "Stratification1": "Sex"},
                      markers=True,
                      color_discrete_map={"Male": SECONDARY, "Female": PRIMARY})
    fig_sex.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          hovermode="x unified", height=360,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig_sex.update_xaxes(showgrid=True, gridcolor="#F0F0F0")
    fig_sex.update_yaxes(showgrid=True, gridcolor="#F0F0F0")

    return fig_inc, fig_edu, fig_race, fig_age, fig_sex


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Starting BRFSS Dashboard on http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
