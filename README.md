# US Health Analytics Dashboard

Interactive data analytics dashboard analyzing obesity, nutrition, and physical activity trends across the United States using CDC public health data.

Author: Himaja Arabati  
GitHub: https://github.com/himaja989

---

## Project Overview

This project analyzes nationwide health patterns using the CDC Behavioral Risk Factor Surveillance System (BRFSS) dataset.  

The goal is to transform raw public health data into meaningful insights through data processing, statistical analysis, and an interactive web-based visualization dashboard.

The final output is a browser-based analytics application that allows users to explore trends across time, geography, and demographics.

---

## Objectives

- Analyze long-term obesity trends in the United States
- Study relationships between physical inactivity and obesity
- Compare nutrition habits across states
- Identify demographic health disparities
- Present insights using an interactive analytics dashboard

---

## Dataset

Source: Centers for Disease Control and Prevention (CDC)  
Dataset: Behavioral Risk Factor Surveillance System (BRFSS)

Dataset includes:
- Adult health survey responses
- State-level indicators
- Multiple years of observations
- Demographic segmentation

Health indicators analyzed:
- Obesity prevalence
- Physical inactivity
- Fruit consumption
- Vegetable consumption

---

## Project Workflow

### 1. Data Collection
- Downloaded CDC public dataset
- Imported into Python environment

### 2. Data Cleaning and Preparation
- Removed missing and suppressed records
- Standardized column names
- Converted data types
- Removed duplicates
- Cleaned categorical variables
- Validated percentage values
- Created analysis-ready dataset

### 3. Exploratory Data Analysis
- Time-series trend analysis
- Geographic comparisons
- Demographic analysis
- Correlation analysis

### 4. Visualization Development
Interactive visualizations created using Plotly:
- Trend line charts
- Choropleth U.S. maps
- Comparative bar charts
- Demographic visualizations

### 5. Dashboard Development
Built a web application using Plotly Dash to provide an interactive analytics experience.

---

## Dashboard Features (Browser Output)

After running the application, the dashboard opens in a browser:

http://127.0.0.1:8050/

Users can:

- Explore obesity trends over multiple years
- Compare health indicators across U.S. states
- Filter data dynamically
- Analyze demographic differences
- View geographic health patterns
- Hover over charts for detailed insights
- Navigate across multiple analytical sections

---

## Key Insights

- Obesity rates show a steady upward national trend
- Physical inactivity strongly correlates with obesity
- Southern states exhibit higher obesity prevalence
- Higher income and education levels associate with healthier outcomes
- Nutrition patterns remain relatively stable over time

---

## Skills Demonstrated

Data Analysis
- Data Cleaning
- Exploratory Data Analysis
- Statistical Interpretation
- Feature Engineering
- Data Storytelling

Data Visualization
- Interactive Dashboard Design
- Geographic Visualization
- Trend Analysis
- Analytical Reporting

Software Development
- End-to-End Data Pipeline
- Dashboard Application Development
- Reproducible Analytics Workflow

---

## Technologies Used

Programming
- Python

Data Processing
- Pandas
- NumPy

Visualization
- Plotly
- Plotly Express

Dashboard Framework
- Dash
- Dash Bootstrap Components

Development Tools
- Jupyter Notebook
- Git
- GitHub

---

## Project Structure

us-health-analytics-dashboard/
│
├── dashboard.py
├── data_processing.ipynb
├── data_processing_executed.ipynb
├── brfss_cleaned.csv
└── README.md


---

## How to Run the Project

### Step 1: Clone Repository

git clone https://github.com/himaja989/us-health-analytics-dashboard.git
cd us-health-analytics-dashboard


### Step 2: Install Dependencies

pip install pandas numpy plotly dash dash-bootstrap-components


### Step 3: Run Dashboard

python dashboard.py


### Step 4: Open Browser
Go to: http://127.0.0.1:8050/


---

## Features Implemented

- Interactive analytics dashboard
- Multi-page navigation
- Dynamic filtering
- Geographic choropleth mapping
- Trend visualization
- Demographic comparison analysis
- Responsive browser interface

---

## Future Improvements

- Cloud deployment
- Real-time data integration
- Machine learning prediction models
- Performance optimization

---

## Contact

Himaja Arabati  
GitHub: https://github.com/himaja989
