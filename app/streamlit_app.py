
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

    import plotly.express as px

    st.title("🚲 Urban Mobility Demand Intelligence")

    st.markdown(
        "Explore historical bike-sharing demand patterns "
        "and machine learning predictions."
    )

    # KPI metrics
    total_records = len(df)
    average_demand = df["cnt"].mean()
    maximum_demand = df["cnt"].max()
    total_rentals = df["cnt"].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Average Hourly Demand", f"{average_demand:.0f}")
    col3.metric("Peak Hourly Demand", f"{maximum_demand:,}")
    col4.metric("Total Rentals", f"{total_rentals:,}")

    st.divider()

    # Hourly demand
    st.subheader("Hourly Rental Demand")

    hourly_demand = (
        df.groupby("hr")["cnt"]
        .mean()
        .reset_index()
    )

    fig_hourly = px.line(
        hourly_demand,
        x="hr",
        y="cnt",
        markers=True,
        title="Average Bike Rentals by Hour"
    )

    fig_hourly.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Average Rentals",
        hovermode="x unified"
    )

    st.plotly_chart(fig_hourly, use_container_width=True)

    # Monthly demand
    st.subheader("Monthly Rental Demand")

    monthly_demand = (
        df.groupby("mnth")["cnt"]
        .mean()
        .reset_index()
    )

    month_names = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    monthly_demand["Month"] = monthly_demand["mnth"].map(
        lambda month: month_names[month - 1]
    )

    fig_monthly = px.bar(
        monthly_demand,
        x="Month",
        y="cnt",
        title="Average Bike Rentals by Month"
    )

    fig_monthly.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Rentals"
    )

    st.plotly_chart(fig_monthly, use_container_width=True)

    # Dataset preview
    st.subheader("Dataset Preview")

    st.caption(
        "A sample of the historical hourly bike-sharing records."
    )

    st.dataframe(
        df.head(10),
        width="stretch"
    )


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
    max_value=57,
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

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Estimated Bike Rentals",
            f"{prediction:,.0f} bikes"
        )

        result_col2.metric(
            "Selected Date",
            date.strftime("%d %b %Y")
        )

        result_col3.metric(
            "Selected Hour",
            f"{selected_hour:02d}:00"
        )

        st.caption(
            "This is a machine learning estimate based on historical "
            "2011–2012 data, not a guaranteed number of rentals."
        )

        st.subheader("Prediction Input Summary")

        season_names = {
            1: "Winter",
            2: "Spring",
            3: "Summer",
            4: "Fall"
        }

        summary = pd.DataFrame({
            "Feature": [
                "Date",
                "Season",
                "Year",
                "Month",
                "Hour",
                "Holiday",
                "Weekday",
                "Working Day",
                "Weather",
                "Temperature",
                "Feels-like Temperature",
                "Humidity",
                "Wind Speed"
            ],
            "Selected Value": [
                date.strftime("%d %B %Y"),
                season_names[season],
                date.year,
                date.strftime("%B"),
                f"{selected_hour:02d}:00",
                "Yes" if holiday else "No",
                date.strftime("%A"),
                "Yes" if workingday else "No",
                selected_weather,
                f"{temperature} °C",
                f"{feels_like} °C",
                f"{humidity}%",
                f"{wind_speed} km/h"
            ]
        })

        st.dataframe(
    summary,
    hide_index=True,
    width="stretch",
    height=500
)


# --------------------------------------------------
# DEMAND ANALYTICS PAGE
# --------------------------------------------------

elif page == "Demand Analytics":

    import plotly.express as px

    st.title("📊 Demand Analytics")

    st.write(
        "Explore the relationship between time, season, "
        "weather and hourly bike rental demand."
    )

    # Demand by Hour
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demand by Hour")

        hourly = (
            df.groupby("hr")["cnt"]
            .mean()
            .reindex(range(24))
            .reset_index()
        )

        fig_hourly = px.bar(
            hourly,
            x="hr",
            y="cnt",
            title="Average Rentals by Hour",
            labels={
                "hr": "Hour of Day",
                "cnt": "Average Rentals"
            }
        )

        st.plotly_chart(fig_hourly, use_container_width=True)

    # Demand by Season
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
            .rename_axis("Season")
            .reset_index(name="Average Rentals")
        )

        fig_season = px.bar(
            seasonal,
            x="Season",
            y="Average Rentals",
            title="Average Rentals by Season"
        )

        st.plotly_chart(fig_season, use_container_width=True)

    st.divider()

    # Demand by Weather
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
        .rename_axis("Weather Condition")
        .reset_index(name="Average Rentals")
    )

    fig_weather = px.bar(
        weather_demand,
        x="Weather Condition",
        y="Average Rentals",
        title="Average Rentals by Weather Condition"
    )

    st.plotly_chart(fig_weather, use_container_width=True)

    # Working Day vs Non-working Day
    st.subheader("Working Day vs Non-working Day")

    working_demand = (
        df.groupby("workingday")["cnt"]
        .mean()
        .rename({
            0: "Non-working Day",
            1: "Working Day"
        })
        .rename_axis("Day Type")
        .reset_index(name="Average Rentals")
    )

    fig_working = px.bar(
        working_demand,
        x="Day Type",
        y="Average Rentals",
        title="Average Rentals by Day Type"
    )

    st.plotly_chart(fig_working, use_container_width=True)

    # Demand Statistics
    st.divider()

    st.subheader("Demand Statistics")

    statistics = (
        df["cnt"]
        .describe()
        .to_frame(name="Hourly Rentals")
        .round(2)
    )

    st.dataframe(statistics, width="stretch")

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

    import plotly.express as px
    import plotly.graph_objects as go

    comparison = pd.DataFrame({
        "Actual Demand": y_test.values,
        "Predicted Demand": predictions
    })

    comparison["Prediction Error"] = (
        comparison["Actual Demand"] -
        comparison["Predicted Demand"]
    )

    fig = px.scatter(
        comparison,
        x="Actual Demand",
        y="Predicted Demand",
        hover_data=["Prediction Error"],
        opacity=0.6,
        title="Actual vs Predicted Bike Rental Demand"
    )

    min_value = min(
        comparison["Actual Demand"].min(),
        comparison["Predicted Demand"].min()
    )

    max_value = max(
        comparison["Actual Demand"].max(),
        comparison["Predicted Demand"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Perfect Prediction",
            line=dict(dash="dash", width=2)
        )
    )

    fig.update_layout(
        xaxis_title="Actual Demand (bikes)",
        yaxis_title="Predicted Demand (bikes)",
        height=600,
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)

        # Residual analysis
    st.subheader("Prediction Error Distribution")

    import matplotlib.pyplot as plt

    residuals = y_test.values - predictions

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(residuals, bins=40, edgecolor="black")

    ax.axvline(
        0,
        linestyle="--",
        linewidth=2,
        label="Zero Error"
    )

    ax.set_title("Distribution of Prediction Errors")
    ax.set_xlabel("Prediction Error (Actual - Predicted)")
    ax.set_ylabel("Number of Predictions")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

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