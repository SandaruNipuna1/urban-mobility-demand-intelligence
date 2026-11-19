# 🚲 Urban Mobility Demand Intelligence

An end-to-end Data Science project analyzing urban bike-sharing demand, understanding factors influencing rental patterns, and predicting hourly bike rentals using machine learning.

## 📌 Project Overview

Bike-sharing services need to understand how rental demand changes throughout the day, across seasons, and under different weather conditions.

This project investigates historical bike-sharing data to identify demand patterns, perform statistical analysis, develop predictive models, and generate insights that can support operational planning, resource allocation, and data-driven decision-making.

## 🎯 Objectives

- Perform data quality assessment and exploratory data analysis (EDA).
- Identify temporal, seasonal, and weather-related rental demand patterns.
- Investigate relationships between environmental factors and rental demand.
- Conduct statistical analysis and hypothesis testing.
- Develop and compare machine learning regression models.
- Evaluate predictive performance using appropriate regression metrics.
- Identify influential features affecting hourly rental demand.
- Build an interactive application for demand prediction and visualization.

## 📊 Dataset

**Source:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

The dataset contains bike-sharing rental records from 2011–2012, including hourly and daily rental information, weather conditions, seasonal indicators, and calendar-related variables.

**Dataset files:**
- `hour.csv` — Hourly bike rental records.
- `day.csv` — Daily bike rental records.

The hourly dataset is used for demand prediction.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Statistical Analysis | SciPy, Statsmodels |
| Machine Learning | Scikit-learn |
| Development | Jupyter Notebook, Git, GitHub |
| Dashboard (planned) | Streamlit |

## 📁 Project Structure

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

## 🔬 Project Workflow

### 1. Data Audit
Inspect dataset structure, data types, missing values, duplicates, and overall data quality.

### 2. Exploratory Data Analysis
Analyze rental demand patterns across hours, weekdays, seasons, and weather conditions.

### 3. Statistical Analysis
Investigate relationships between variables and conduct statistical hypothesis tests.

### 4. Machine Learning
Develop and compare regression models for hourly rental demand prediction.

**Models explored:**
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

## 📈 Models

Machine learning regression models are developed and compared to predict hourly bike rental demand.

Model development includes baseline comparison, model evaluation, and hyperparameter tuning.

## 📏 Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## 🚧 Project Progress

- [x] Data Audit
- [x] Exploratory Data Analysis
- [x] Statistical Analysis
- [x] Model Development and Evaluation
- [ ] Final Model Refinement
- [ ] Interactive Streamlit Dashboard
- [ ] Deployment

## 🔁 Reproducibility

The dataset can be downloaded from the UCI Machine Learning Repository using the link above.

Raw data files and generated model artifacts are excluded from version control. The notebooks document the analysis and model development workflow.

## 👨‍💻 Author

**Sandaru Nipuna Nethsara Nanayakkara**

GitHub: [SandaruNipuna1](https://github.com/SandaruNipuna1)