# 🚲 Urban Mobility Demand Intelligence

An end-to-end Data Science and Machine Learning project that analyzes urban bike-sharing demand, discovers temporal and environmental patterns, and predicts hourly bike rentals using machine learning.

🌐 **Live Application:** [Urban Mobility Demand Intelligence](https://urban-mobility-demand-intelligence-z5falbkevp6lij56g873rv.streamlit.app/)

💻 **GitHub Repository:** [urban-mobility-demand-intelligence](https://github.com/SandaruNipuna1/urban-mobility-demand-intelligence)

---

## 📌 Project Overview

Bike-sharing services need to understand how rental demand changes throughout the day, across seasons, and under different weather conditions.

This project uses historical bike-sharing data to investigate demand patterns, perform statistical analysis, develop predictive machine learning models, and deliver an interactive application for demand exploration and prediction.

The project demonstrates a complete Data Science workflow, from data quality assessment and exploratory analysis to model development, evaluation, dashboard development, and cloud deployment.

## 🎯 Objectives

- Perform data quality assessment and exploratory data analysis (EDA).
- Identify temporal, seasonal, and weather-related rental demand patterns.
- Investigate relationships between environmental factors and rental demand.
- Conduct statistical analysis and hypothesis testing.
- Develop and compare machine learning regression models.
- Evaluate predictive performance using appropriate regression metrics.
- Identify influential features affecting hourly rental demand.
- Build an interactive application for demand prediction and visualization.
- Deploy the application to a cloud hosting platform.

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
| Visualization | Matplotlib, Seaborn, Plotly |
| Statistical Analysis | SciPy, Statsmodels |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Dashboard | Streamlit |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## 📁 Project Structure

```text
urban-mobility-demand-intelligence/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── hour.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_statistical_analysis.ipynb
│   ├── 04_model_development.ipynb
│   └── best_random_forest_model_compressed.joblib
│
├── src/
│
├── reports/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔬 Project Workflow

### 1. Data Audit

- Inspected dataset structure and dimensions.
- Examined data types and missing values.
- Checked duplicate records.
- Assessed overall data quality.

### 2. Exploratory Data Analysis

Investigated bike rental demand patterns across:

- Hours of the day.
- Months and seasons.
- Working days and non-working days.
- Weather conditions.
- Temporal demand variations.

### 3. Statistical Analysis

- Investigated relationships between environmental variables and rental demand.
- Conducted statistical analysis and hypothesis testing.
- Examined statistical significance and variable relationships.

### 4. Machine Learning

Developed and compared multiple regression models to predict hourly bike rental demand.

**Models explored:**

- Dummy Regressor (Baseline)
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Tuned Random Forest Regressor

A chronological 80/20 train-test split was used to evaluate predictive performance on later observations.

TimeSeriesSplit and GridSearchCV were used for hyperparameter optimization.

Feature leakage was avoided by excluding `casual`, `registered`, and the target variable `cnt` from the model inputs.

### 5. Model Evaluation

Models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

**Best-performing model: Random Forest Regressor**

| Evaluation Metric | Test Result |
|---|---:|
| MAE | 45.09 |
| RMSE | 69.69 |
| R² Score | 0.9001 |

The model achieved an R² score of approximately 0.90 on the chronological held-out test set.

Additional evaluation included:

- Actual vs. predicted demand analysis.
- Residual distribution analysis.
- Prediction error investigation.
- Feature importance analysis.

### 6. Interactive Dashboard

Developed an interactive Streamlit application with four main sections.

**Overview**
- Dataset summary and key performance indicators.
- Hourly rental demand visualization.
- Monthly demand analysis.
- Dataset preview.

**Demand Prediction**
- Interactive date and time selection.
- Weather and environmental inputs.
- Real-time hourly bike rental demand prediction.
- Prediction input summary.

**Demand Analytics**
- Demand by hour.
- Demand by season.
- Demand by weather condition.
- Working day vs. non-working day analysis.

**Model Performance**
- MAE, RMSE, and R² metrics.
- Actual vs. predicted demand visualization.
- Prediction error distribution.
- Model evaluation explanations.

### 7. Model Optimization and Deployment

- Compressed the trained Random Forest model using Joblib.
- Reduced model artifact size from approximately 46 MB to 11 MB.
- Verified prediction equivalence between the original and compressed models.
- Integrated the compressed model into the Streamlit application.
- Published the application using Streamlit Community Cloud.
- Managed project development and version control through GitHub.

## 🌐 Live Application

**Explore the deployed application:**

[🚀 Open Urban Mobility Demand Intelligence](https://urban-mobility-demand-intelligence-z5falbkevp6lij56g873rv.streamlit.app/)

The application provides interactive visualizations, historical demand exploration, machine learning predictions, and model evaluation insights.

## 📏 Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Measures average absolute prediction error. |
| RMSE | Measures prediction error while penalizing larger errors more heavily. |
| R² Score | Measures the proportion of target variance explained by the model. |

## ⚙️ Installation and Usage

Clone the repository:

```bash
git clone https://github.com/SandaruNipuna1/urban-mobility-demand-intelligence.git
```

Navigate to the project directory:

```bash
cd urban-mobility-demand-intelligence
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

## 🔁 Reproducibility

The project includes Jupyter notebooks documenting the data audit, exploratory analysis, statistical analysis, and model development workflow.

The compressed trained model and required hourly dataset are included in the repository to support application execution.

The original UCI dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

## 🚀 Project Status

- [x] Data Audit
- [x] Exploratory Data Analysis
- [x] Statistical Analysis
- [x] Machine Learning Model Development
- [x] Model Comparison and Hyperparameter Tuning
- [x] Model Evaluation
- [x] Model Compression and Optimization
- [x] Interactive Streamlit Dashboard
- [x] GitHub Repository
- [x] Cloud Deployment

**Project Status: Completed and Deployed** 🎉

## 👨‍💻 Author

**Sandaru Nipuna Nethsara Nanayakkara**

BSc (Hons) Information Technology – Data Science

GitHub: [SandaruNipuna1](https://github.com/SandaruNipuna1)

---

⭐ If you find this project interesting, feel free to explore the repository and the live application.
