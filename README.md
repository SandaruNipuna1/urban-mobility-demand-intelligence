# 🚲 Urban Mobility Demand Intelligence

An end-to-end Data Science project analyzing urban bike-sharing demand and developing machine learning models to predict hourly rental demand.

## 📌 Project Overview

Bike-sharing services need to understand how rental demand changes throughout the day, across seasons, and under different weather conditions.

This project uses historical bike-sharing data to uncover demand patterns, investigate factors influencing rentals, conduct statistical analysis, and develop machine learning models for demand prediction. The findings can support better resource allocation, operational planning, and data-driven decision-making.

## 🎯 Objectives

- Perform data quality assessment and exploratory data analysis (EDA).
- Identify temporal, seasonal, and weather-related rental demand patterns.
- Investigate relationships between environmental factors and rental demand.
- Conduct statistical analysis and hypothesis testing.
- Develop and compare machine learning regression models.
- Evaluate models using appropriate regression metrics.
- Build an interactive application for demand prediction and visualization.

## 📊 Dataset

**Source:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

The dataset contains bike-sharing rental records from 2011–2012, including hourly and daily rental information, weather conditions, seasonal indicators, and calendar-related variables.

**Dataset files:**
- `hour.csv` — Hourly bike rental records.
- `day.csv` — Daily bike rental records.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Statistical Analysis | SciPy, Statsmodels |
| Machine Learning | Scikit-learn |
| Development | Jupyter Notebook, Git, GitHub |
| Deployment (planned) | Streamlit |

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

1. **Data Audit** — Inspect data structure, missing values, duplicates, data types, and quality.
2. **Exploratory Data Analysis** — Explore rental patterns, trends, distributions, and relationships.
3. **Statistical Analysis** — Conduct statistical tests and investigate significant relationships.
4. **Model Development** — Train, compare, and evaluate machine learning regression models.
5. **Interactive Application** *(planned)* — Develop a dashboard for demand insights and interactive predictions.

## 📈 Models

Machine learning regression models are developed and compared to predict hourly bike rental demand.

Models explored include:

- Dummy Regression (Baseline)
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Hyperparameter tuning and model optimization

## 📏 Evaluation Metrics

Models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## 🚧 Project Status

**Currently in development.**

- [x] Data Audit
- [x] Exploratory Data Analysis
- [x] Statistical Analysis
- [x] Model Development & Evaluation
- [ ] Final model refinement
- [ ] Interactive Streamlit dashboard
- [ ] Deployment

## 👨‍💻 Author

**Sandaru Nipuna Nethsara Nanayakkara**

GitHub: [SandaruNipuna1](https://github.com/SandaruNipuna1)
