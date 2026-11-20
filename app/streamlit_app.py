
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Urban Mobility Demand Intelligence",
    page_icon="🚲",
    layout="wide"
)

# --------------------------------------------------
# PROJECT PATHS AND MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "notebooks" / "best_random_forest_model.joblib"
DATA_PATH = BASE_DIR / "data" / "raw" / "hour.csv"

FEATURES = [
    "season", "yr", "mnth", "hr", "holiday",
    "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed"
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["dteday"] = pd.to_datetime(df["dteday"])
    return df


model = load_model()
df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🚲 Urban Mobility")
st.sidebar.caption("Demand Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
        [
        "Overview",
        "Demand Prediction",
        "Demand Analytics",
        "Model Performance"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Data source: UCI Bike Sharing Dataset")
st.sidebar.caption("Model: Random Forest Regressor")

# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    st.title("Urban Mobility Demand Intelligence")
    st.subheader("Hourly Bike Rental Demand Analysis")

    st.write(
        "Explore historical bike rental patterns and predict "
        "hourly demand using a machine learning model."
    )

    total_records = len(df)
    average_demand = df["cnt"].mean()
    maximum_demand = df["cnt"].max()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Hourly Records", f"{total_records:,}")
    col2.metric("Average Hourly Rentals", f"{average_demand:.0f}")
    col3.metric("Maximum Hourly Rentals", f"{maximum_demand:,}")

    st.divider()

    st.subheader("Historical Hourly Demand")

    hourly_demand = (
        df.groupby("hr")["cnt"]
        .mean()
        .reindex(range(24))
    )

    st.line_chart(hourly_demand)

    st.subheader("Monthly Rental Demand")

    monthly_demand = (
        df.groupby("mnth")["cnt"]
        .mean()
        .reindex(range(1, 13))
    )

    st.bar_chart(monthly_demand)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)


# --------------------------------------------------
# DEMAND PREDICTION PAGE
# --------------------------------------------------

elif page == "Demand Prediction":

    st.title("🚲 Hourly Demand Prediction")

    st.write(
        "Configure the date, time and weather conditions "
        "to estimate the expected number of bike rentals."
    )

    st.info(
        "This model was trained on historical data from 2011–2012. "
        "Predictions are estimates based on patterns learned from "
        "that dataset."
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.date_input(
            "Select Date",
            value=pd.Timestamp("2012-06-15").date(),
            min_value=pd.Timestamp("2011-01-01").date(),
            max_value=pd.Timestamp("2012-12-31").date()
        )

        selected_hour = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=8
        )

        holiday = st.checkbox("Public holiday")

    with col2:
        weather_options = {
            "Clear / Few clouds": 1,
            "Mist / Cloudy": 2,
            "Light rain / Light snow": 3,
            "Heavy rain / Snow": 4
        }

        selected_weather = st.selectbox(
            "Weather Condition",
            list(weather_options.keys())
        )

        temperature = st.slider(
            "Temperature (°C)",
            min_value=0,
            max_value=41,
            value=20
        )

        feels_like = st.slider(
            "Feels-like Temperature (°C)",
            min_value=0,
            max_value=50,
            value=22
        )

    col3, col4 = st.columns(2)

    with col3:
        humidity = st.slider(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=60
        )

    with col4:
        wind_speed = st.slider(
            "Wind Speed (km/h)",
            min_value=0,
            max_value=67,
            value=15
        )

    # Date-based feature engineering

    date = pd.Timestamp(selected_date)

    month = date.month
    year = date.year
    weekday = (date.weekday() + 1) % 7

    # UCI season encoding:
    # 1 = Winter, 2 = Spring, 3 = Summer, 4 = Fall

    # UCI season encoding based on dataset transition dates

    if (month == 12 and date.day >= 21) or month in [1, 2] or (month == 3 and date.day <= 20):
        season = 1
    elif (month == 3 and date.day >= 21) or month in [4, 5] or (month == 6 and date.day <= 20):
        season = 2
    elif (month == 6 and date.day >= 21) or month in [7, 8] or (month == 9 and date.day <= 22):
        season = 3
    else:
        season = 4

    yr = year - 2011

    workingday = int(weekday not in [0, 6] and not holiday)

    # Convert real-world inputs to dataset-normalized values

    normalized_temp = temperature / 41
    normalized_atemp = feels_like / 50
    normalized_humidity = humidity / 100
    normalized_windspeed = wind_speed / 67

    input_data = pd.DataFrame(
        [[
            season,
            yr,
            month,
            selected_hour,
            int(holiday),
            weekday,
            workingday,
            weather_options[selected_weather],
            normalized_temp,
            normalized_atemp,
            normalized_humidity,
            normalized_windspeed
        ]],
        columns=FEATURES
    )

    st.divider()

    if st.button("Predict Bike Rental Demand", type="primary"):

        prediction = model.predict(input_data)[0]
        prediction = max(0, prediction)

        st.success("Prediction completed successfully!")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Estimated Bike Rentals",
            f"{prediction:,.0f}"
        )

        result_col2.metric(
            "Selected Hour",
            f"{selected_hour:02d}:00"
        )

        st.caption(
            "This is a machine learning estimate, not a guaranteed "
            "number of rentals."
        )

        st.subheader("Prediction Input Summary")
        st.dataframe(input_data, use_container_width=True)


# --------------------------------------------------
# DEMAND ANALYTICS PAGE
# --------------------------------------------------

elif page == "Demand Analytics":

    st.title("📊 Demand Analytics")

    st.write(
        "Explore the relationship between time, season, "
        "weather and hourly bike rental demand."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demand by Hour")

        hourly = (
            df.groupby("hr")["cnt"]
            .mean()
            .reindex(range(24))
        )

        st.bar_chart(hourly)

    with col2:
        st.subheader("Demand by Season")

        seasonal = (
            df.groupby("season")["cnt"]
            .mean()
            .rename({
                1: "Winter",
                2: "Spring",
                3: "Summer",
                4: "Fall"
            })
        )

        st.bar_chart(seasonal)

    st.subheader("Demand by Weather Condition")

    weather_demand = (
        df.groupby("weathersit")["cnt"]
        .mean()
        .rename({
            1: "Clear",
            2: "Mist / Cloudy",
            3: "Light Rain / Snow",
            4: "Heavy Rain / Snow"
        })
    )

    st.bar_chart(weather_demand)

    st.subheader("Working Day vs Non-working Day")

    working_demand = (
        df.groupby("workingday")["cnt"]
        .mean()
        .rename({
            0: "Non-working Day",
            1: "Working Day"
        })
    )

    st.bar_chart(working_demand)

    st.subheader("Demand Statistics")

    st.dataframe(
        df["cnt"].describe().to_frame().round(2),
        use_container_width=True
    )

# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("🎯 Model Performance")
    st.write(
        "Evaluate the Random Forest model using "
        "chronologically held-out test data."
    )

    # Prepare chronological test set
    split_index = int(len(df) * 0.8)
    test_data = df.iloc[split_index:].copy()

    X_test = test_data[FEATURES]
    y_test = test_data["cnt"]

    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )
    import numpy as np

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Display metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:.2f} bikes")
    col2.metric("RMSE", f"{rmse:.2f} bikes")
    col3.metric("R² Score", f"{r2:.4f}")

    st.divider()

    # Actual vs predicted
    st.subheader("Actual vs Predicted Demand")

    comparison = pd.DataFrame({
        "Actual Demand": y_test.values,
        "Predicted Demand": predictions
    })

    st.scatter_chart(
        comparison,
        x="Actual Demand",
        y="Predicted Demand"
    )

    # Residual analysis
    st.subheader("Prediction Error Distribution")

    residuals = y_test.values - predictions

    st.histogram = None
    residual_data = pd.DataFrame({
        "Prediction Error (Actual - Predicted)": residuals
    })

    st.bar_chart(
        residual_data["Prediction Error (Actual - Predicted)"]
        .value_counts(bins=30)
        .sort_index()
    )

    # Explain metrics
    st.subheader("Understanding the Results")

    st.write(
        f"""
        - **MAE ({mae:.2f}):** On average, predictions differ
          from actual demand by approximately {mae:.0f} bikes.
        - **RMSE ({rmse:.2f}):** Measures prediction error while
          penalizing larger errors more heavily.
        - **R² ({r2:.4f}):** The model explains approximately
          {r2 * 100:.2f}% of the variation in hourly demand.
        """
    )

    st.caption(
        "Evaluation uses the final 20% of records in chronological "
        "order, not the training dataset."
    )