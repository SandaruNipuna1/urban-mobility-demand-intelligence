# 🚲 Urban Mobility Demand Intelligence

An end-to-end Data Science project focused on analyzing urban bike-sharing patterns, understanding factors influencing rental demand, and predicting hourly bike rentals using machine learning.

## Project Overview

Bike-sharing services need to understand how rental demand changes throughout the day and across different weather and seasonal conditions.

This project investigates historical bike-sharing data to identify demand patterns, perform statistical analysis, develop predictive models, and generate insights that can support operational planning and resource allocation.

## Objectives

- Perform data quality assessment and exploratory data analysis.
- Investigate relationships between weather, time, and rental demand.
- Conduct statistical analysis and hypothesis testing.
- Develop and compare machine learning regression models.
- Evaluate predictive performance using appropriate regression metrics.
- Identify influential features affecting hourly rental demand.
- Build an interactive application to demonstrate predictions.

## Dataset

**Source:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

The dataset contains hourly and daily bike rental information from 2011–2012, alongside weather and calendar-related variables.

The hourly dataset is used for demand prediction.

## Technology Stack

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Statistical Analysis:** SciPy, Statsmodels
- **Machine Learning:** Scikit-learn
- **Development:** Jupyter Notebook, Git, GitHub
- **Dashboard (planned):** Streamlit

## Project Structure

```text
urban-mobility-demand-intelligence/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_statistical_analysis.ipynb
│   └── 04_model_development.ipynb
├── src/
├── app/
├── reports/
├── requirements.txt
└── README.md
```

## Project Workflow

### 1. Data Audit
Inspect dataset structure, data types, missing values, duplicates, and overall data quality.

### 2. Exploratory Data Analysis
Analyze rental demand patterns across hours, weekdays, seasons, and weather conditions.

### 3. Statistical Analysis
Investigate relationships between variables and conduct statistical hypothesis tests.

### 4. Machine Learning
Develop and compare regression models for hourly rental demand prediction.

Models explored:
- Dummy Regressor (baseline)
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Tuned Random Forest

A chronological train-test split is used to evaluate predictive performance on later observations.

### 5. Model Evaluation
Evaluate models using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Additional analysis includes feature importance, actual versus predicted values, residual analysis, and prediction errors by hour.

### 6. Interactive Dashboard (Planned)
Develop a Streamlit application to visualize demand patterns and demonstrate model predictions.

## Project Progress

- [x] Data Audit
- [x] Exploratory Data Analysis
- [x] Statistical Analysis
- [x] Model Development and Evaluation
- [ ] Final Model Refinement
- [ ] Interactive Streamlit Dashboard
- [ ] Deployment

## Reproducibility

The dataset can be downloaded from the UCI Machine Learning Repository using the link above.

Raw data files and generated model artifacts are excluded from version control. The notebooks document the analysis and model development workflow.

## Author

**Sandaru Nipuna Nethsara Nanayakkara**

[GitHub](https://github.com/SandaruNipuna1)
