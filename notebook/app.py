
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="WeatherWise",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# 2. HEADER
# =====================================================

st.title("🌦️ WeatherWise")

st.caption(
    "Machine Learning Based Weather Prediction "
    "& Analysis System"
)


# =====================================================
# 3. CITY COORDINATES
# =====================================================

cities = {
    "Dhaka": (23.8103, 90.4125),
    "Chattogram": (22.3569, 91.7832),
    "Rajshahi": (24.3745, 88.6042),
    "Khulna": (22.8456, 89.5403),
    "Sylhet": (24.8949, 91.8687),
    "Barishal": (22.7010, 90.3535),
    "Rangpur": (25.7439, 89.2752),
    "Mymensingh": (24.7471, 90.4203)
}


FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "pressure_msl",
    "cloud_cover",
    "Hour",
    "Month",
    "City"
]


CLUSTER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "pressure_msl",
    "cloud_cover"
]


# =====================================================
# 4. LIVE WEATHER API
# =====================================================

@st.cache_data(ttl=600)
def get_live_weather(city):

    latitude, longitude = cities[city]

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "rain,"
            "pressure_msl,"
            "wind_speed_10m,"
            "cloud_cover"
        ),
        "timezone": "Asia/Dhaka"
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return response.json()["current"]


# =====================================================
# 5. WEATHER METRICS
# =====================================================

def show_weather_metrics(current):

    st.subheader("Current Weather Information")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🌡️ Temperature",
        f"{current['temperature_2m']} °C"
    )

    c2.metric(
        "💧 Humidity",
        f"{current['relative_humidity_2m']}%"
    )

    c3.metric(
        "💨 Wind Speed",
        f"{current['wind_speed_10m']} km/h"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🌧️ Rain",
        f"{current['rain']} mm"
    )

    c2.metric(
        "☁️ Cloud Cover",
        f"{current['cloud_cover']}%"
    )

    c3.metric(
        "Pressure",
        f"{current['pressure_msl']} hPa"
    )


# =====================================================
# 6. LOAD HISTORICAL DATA
# =====================================================

@st.cache_data
def load_data():

    file_path = (
        Path(__file__).resolve().with_name(
            "bangladesh_historical_weather.csv"
        )
    )

    df = pd.read_csv(
        file_path,
        parse_dates=["time"]
    )

    df = df.sort_values(
        ["City", "time"]
    ).copy()

    df["Hour"] = df["time"].dt.hour

    df["Month"] = df["time"].dt.month

    df["Next_Hour_Temp"] = (
        df.groupby("City")["temperature_2m"].shift(-1)
    )

    df["Next_Hour_Rain"] = (
        df.groupby("City")["rain"].shift(-1)
    )

    df["Rain_Target"] = (
        df["Next_Hour_Rain"] > 0
    ).astype(int)

    df = df.dropna(
        subset=FEATURES + [
            "Next_Hour_Temp",
            "Next_Hour_Rain"
        ]
    ).copy()

    return df


# =====================================================
# 7. TRAIN MACHINE LEARNING MODELS
# =====================================================

@st.cache_resource
def train_models():

    df = load_data()

    # Chronological Train-Test Split

    cutoff = df.groupby("City")["time"].transform(
        lambda x: x.quantile(0.8)
    )

    train_mask = df["time"] <= cutoff

    train = df.loc[train_mask].copy()

    test = df.loc[~train_mask].copy()

    # One Hot Encoding

    X_train = pd.get_dummies(
        train[FEATURES],
        columns=["City"],
        dtype=int
    )

    X_test = pd.get_dummies(
        test[FEATURES],
        columns=["City"],
        dtype=int
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # ---------------------------------
    # Linear Regression
    # ---------------------------------

    temp_model = LinearRegression()

    temp_model.fit(
        X_train,
        train["Next_Hour_Temp"]
    )

    temp_predictions = temp_model.predict(
        X_test
    )

    # ---------------------------------
    # Logistic Regression
    # ---------------------------------

    rain_scaler = StandardScaler()

    X_train_scaled = rain_scaler.fit_transform(
        X_train
    )

    X_test_scaled = rain_scaler.transform(
        X_test
    )

    rain_model = LogisticRegression(
        max_iter=1000
    )

    rain_model.fit(
        X_train_scaled,
        train["Rain_Target"]
    )

    rain_predictions = rain_model.predict(
        X_test_scaled
    )

    # ---------------------------------
    # K-Means Clustering
    # ---------------------------------

    cluster_scaler = StandardScaler()

    cluster_train = cluster_scaler.fit_transform(
        train[CLUSTER_FEATURES]
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    sample_size = min(
        50000,
        len(cluster_train)
    )

    rng = np.random.default_rng(42)

    sample_indices = rng.choice(
        len(cluster_train),
        size=sample_size,
        replace=False
    )

    kmeans.fit(
        cluster_train[sample_indices]
    )

    # ---------------------------------
    # LDA + Logistic Regression
    # ---------------------------------

    lda = LinearDiscriminantAnalysis(
        n_components=1
    )

    lda_sample_size = min(
        50000,
        len(X_train_scaled)
    )

    lda_indices = rng.choice(
        len(X_train_scaled),
        size=lda_sample_size,
        replace=False
    )

    lda_X = X_train_scaled[lda_indices]

    lda_y = train["Rain_Target"].iloc[
        lda_indices
    ]

    lda.fit(
        lda_X,
        lda_y
    )

    lda_train = lda.transform(lda_X)

    lda_test = lda.transform(
        X_test_scaled
    )

    lda_model = LogisticRegression(
        max_iter=1000
    )

    lda_model.fit(
        lda_train,
        lda_y
    )

    lda_pred = lda_model.predict(
        lda_test
    )

    # ---------------------------------
    # MODEL EVALUATION
    # ---------------------------------

    metrics = {

        "MAE": mean_absolute_error(
            test["Next_Hour_Temp"],
            temp_predictions
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                test["Next_Hour_Temp"],
                temp_predictions
            )
        ),

        "R2": r2_score(
            test["Next_Hour_Temp"],
            temp_predictions
        ),

        "Rain Accuracy": accuracy_score(
            test["Rain_Target"],
            rain_predictions
        ),

        "Rain Precision": precision_score(
            test["Rain_Target"],
            rain_predictions,
            zero_division=0
        ),

        "Rain Recall": recall_score(
            test["Rain_Target"],
            rain_predictions,
            zero_division=0
        ),

        "Rain F1": f1_score(
            test["Rain_Target"],
            rain_predictions,
            zero_division=0
        ),

        "LDA Accuracy": accuracy_score(
            test["Rain_Target"],
            lda_pred
        )
    }

    cm = confusion_matrix(
        test["Rain_Target"],
        rain_predictions,
        labels=[0, 1]
    )

    return (
        df,
        temp_model,
        rain_model,
        rain_scaler,
        kmeans,
        cluster_scaler,
        X_train.columns.tolist(),
        metrics,
        cm,
        train,
        test
    )


# =====================================================
# 8. LOAD TRAINED MODELS
# =====================================================

try:

    (
        df,
        temp_model,
        rain_model,
        rain_scaler,
        kmeans,
        cluster_scaler,
        feature_columns,
        metrics,
        cm,
        train,
        test

    ) = train_models()

except Exception as e:

    st.error(f"Project error: {e}")

    st.info(
        "Check your CSV file, column names, "
        "data and model training configuration."
    )

    st.stop()


# =====================================================
# 9. NAVIGATION
# =====================================================

DASHBOARDS = {

    "Temperature Prediction": (
        "🌡️",
        "Predict next hour temperature"
    ),

    "Rain Prediction": (
        "🌧️",
        "Predict rain probability"
    ),

    "Weather Clustering": (
        "🌤️",
        "Discover weather patterns"
    ),

    "Statistical Analysis": (
        "📊",
        "Weather statistics and Markov"
    ),

    "Model Evaluation": (
        "📈",
        "Evaluate machine learning models"
    ),

    "Live Weather": (
        "🌍",
        "Live weather of 8 cities"
    )
}


if "page" not in st.session_state:

    st.session_state.page = "Home"


def navigate(page):

    st.session_state.page = page

    st.rerun()


def back_to_home():

    if st.button(
        "← Back to Home",
        key=f"back_{st.session_state.page}"
    ):

        navigate("Home")


page = st.session_state.page


# =====================================================
# 10. HOME DASHBOARD
# =====================================================

if page == "Home":

    st.divider()

    st.header("📊 WeatherWise Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Historical Records",
        f"{len(df):,}"
    )

    c2.metric(
        "Bangladesh Cities",
        df["City"].nunique()
    )

    c3.metric(
        "ML Algorithms",
        "4"
    )

    st.divider()

    # ---------------------------------
    # LIVE WEATHER
    # ---------------------------------

    st.header("🌍 Live Weather")

    selected_city = st.selectbox(
        "Select City",
        list(cities.keys()),
        index=list(cities.keys()).index("Mymensingh"),
        key="home_city"
    )

    try:

        current = get_live_weather(
            selected_city
        )

        st.subheader(
            f"🌤️ Current Weather — {selected_city}, Bangladesh"
        )

        st.caption(
            f"Observation Time: {current['time']}"
        )

        show_weather_metrics(current)

    except Exception as e:

        st.error(
            f"Unable to load live weather: {e}"
        )

    st.divider()

    # ---------------------------------
    # HISTORICAL TEMPERATURE CHART
    # ---------------------------------

    st.header("📉 Historical Temperature Trend")

    fig, ax = plt.subplots(figsize=(12, 4))

    historical = (
        df.groupby("time")["temperature_2m"]
        .mean()
    )

    ax.plot(
        historical.index,
        historical.values,
        linewidth=1.5
    )

    ax.set_xlabel("Time")

    ax.set_ylabel("Temperature (°C)")

    ax.set_title("Average Historical Temperature")

    ax.grid(alpha=0.2)

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # ---------------------------------
    # DASHBOARD GRID
    # ---------------------------------

    st.header("🚀 Explore Dashboards")

    st.write(
        "Select a dashboard to explore different "
        "machine learning functionalities."
    )

    dashboard_names = list(
        DASHBOARDS.keys()
    )

    for start in range(
        0,
        len(dashboard_names),
        3
    ):

        cols = st.columns(3)

        for col, name in zip(
            cols,
            dashboard_names[start:start + 3]
        ):

            icon, description = DASHBOARDS[name]

            with col:

                with st.container(border=True):

                    st.subheader(
                        f"{icon} {name}"
                    )

                    st.caption(description)

                    if st.button(
                        "Explore Dashboard →",
                        key=f"nav_{name}",
                        use_container_width=True
                    ):

                        navigate(name)

    st.divider()

    st.caption(
        "WeatherWise | Machine Learning "
        "Based Weather Prediction & Analysis"
    )


# =====================================================
# 11. TEMPERATURE PREDICTION
# =====================================================

elif page == "Temperature Prediction":

    back_to_home()

    st.header("🌡️ Next Hour Temperature Prediction")

    st.write(
        "Linear Regression based temperature prediction."
    )

    city = st.selectbox(
        "Select City",
        list(cities.keys()),
        key="temp_city"
    )

    city_data = df[
        df["City"] == city
    ].iloc[-1]

    c1, c2 = st.columns(2)

    with c1:

        current_temp = st.number_input(
            "Temperature (°C)",
            value=float(city_data["temperature_2m"])
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(city_data["relative_humidity_2m"])
        )

        wind = st.number_input(
            "Wind Speed (km/h)",
            min_value=0.0,
            value=float(city_data["wind_speed_10m"])
        )

    with c2:

        pressure = st.number_input(
            "Pressure (hPa)",
            value=float(city_data["pressure_msl"])
        )

        cloud = st.number_input(
            "Cloud Cover (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(city_data["cloud_cover"])
        )

        hour = st.slider(
            "Hour",
            0,
            23,
            12
        )

        month = st.slider(
            "Month",
            1,
            12,
            11
        )

    if st.button(
        "Predict Temperature",
        type="primary"
    ):

        input_data = pd.DataFrame([{

            "temperature_2m": current_temp,

            "relative_humidity_2m": humidity,

            "wind_speed_10m": wind,

            "pressure_msl": pressure,

            "cloud_cover": cloud,

            "Hour": hour,

            "Month": month,

            "City": city

        }])

        input_data = pd.get_dummies(
            input_data,
            columns=["City"],
            dtype=int
        )

        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = temp_model.predict(
            input_data
        )[0]

        st.success(
            f"Predicted Next Hour Temperature: "
            f"{prediction:.2f} °C"
        )


# =====================================================
# 12. RAIN PREDICTION
# =====================================================

elif page == "Rain Prediction":

    back_to_home()

    st.header("🌧️ Next Hour Rain Prediction")

    st.write(
        "Logistic Regression based rain prediction."
    )

    city = st.selectbox(
        "Select City",
        list(cities.keys()),
        key="rain_city"
    )

    city_data = df[
        df["City"] == city
    ].iloc[-1]

    values = {}

    c1, c2 = st.columns(2)

    for i, feature in enumerate(
        CLUSTER_FEATURES
    ):

        target_col = c1 if i % 2 == 0 else c2

        with target_col:

            values[feature] = st.number_input(
                feature,
                value=float(city_data[feature]),
                key=f"rain_{feature}"
            )

    hour = st.slider(
        "Hour",
        0,
        23,
        12,
        key="rain_hour"
    )

    month = st.slider(
        "Month",
        1,
        12,
        11,
        key="rain_month"
    )

    if st.button(
        "Predict Rain",
        type="primary"
    ):

        input_data = pd.DataFrame([{

            **values,

            "Hour": hour,

            "Month": month,

            "City": city

        }])

        input_data = pd.get_dummies(
            input_data,
            columns=["City"],
            dtype=int
        )

        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        scaled = rain_scaler.transform(
            input_data
        )

        prediction = rain_model.predict(
            scaled
        )[0]

        probability = rain_model.predict_proba(
            scaled
        )[0][1]

        if prediction == 1:

            st.warning(
                "🌧️ Rain is predicted for the next hour."
            )

        else:

            st.success(
                "☀️ No rain is predicted for the next hour."
            )

        st.metric(
            "Predicted Rain Probability",
            f"{probability * 100:.2f}%"
        )


# =====================================================
# 13. WEATHER CLUSTERING
# =====================================================

elif page == "Weather Clustering":

    back_to_home()

    st.header("🌤️ Weather Pattern Clustering")

    st.write(
        "K-Means Clustering with K = 4"
    )

    st.metric(
        "Number of Clusters",
        4
    )

    st.subheader(
        "Cluster Centroids (Scaled Features)"
    )

    st.dataframe(
        pd.DataFrame(
            kmeans.cluster_centers_,
            columns=CLUSTER_FEATURES
        ),
        use_container_width=True
    )

    city = st.selectbox(
        "Select City",
        list(cities.keys()),
        key="cluster_city"
    )

    city_data = df[
        df["City"] == city
    ].iloc[-1]

    cluster_input = (
        city_data[CLUSTER_FEATURES]
        .to_frame()
        .T
    )

    scaled_input = cluster_scaler.transform(
        cluster_input
    )

    cluster = kmeans.predict(
        scaled_input
    )[0]

    st.success(
        f"{city}'s weather belongs to Cluster {cluster}"
    )


# =====================================================
# 14. STATISTICAL ANALYSIS
# =====================================================

elif page == "Statistical Analysis":

    back_to_home()

    st.header("📊 Statistical Analysis")

    # ---------------------------------
    # TEMPERATURE DISTRIBUTION
    # ---------------------------------

    st.subheader(
        "Temperature Probability Density"
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(
        df["temperature_2m"].dropna(),
        bins=50,
        density=True,
        alpha=0.8
    )

    ax.set_xlabel(
        "Temperature (°C)"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Temperature Distribution"
    )

    ax.grid(alpha=0.2)

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # ---------------------------------
    # TEMPERATURE STATISTICS
    # ---------------------------------

    st.subheader(
        "Temperature Statistics"
    )

    st.dataframe(
        df["temperature_2m"].describe(),
        use_container_width=True
    )

    st.divider()

    # ---------------------------------
    # MARKOV TRANSITION MATRIX
    # ---------------------------------

    st.subheader(
        "Markov Transition Matrix"
    )

    states = df.copy()

    states["Weather_State"] = (
        states["rain"] > 0
    ).astype(int)

    states["Next_State"] = (
        states.groupby("City")[
            "Weather_State"
        ].shift(-1)
    )

    counts = pd.crosstab(
        states["Weather_State"],
        states["Next_State"]
    )

    counts = counts.reindex(
        index=[0, 1],
        columns=[0, 1],
        fill_value=0
    )

    transition = counts.div(
        counts.sum(axis=1).replace(0, np.nan),
        axis=0
    )

    transition.index = [
        "No Rain",
        "Rain"
    ]

    transition.columns = [
        "No Rain",
        "Rain"
    ]

    st.dataframe(
        transition.style.format("{:.4f}"),
        use_container_width=True
    )


# =====================================================
# 15. MODEL EVALUATION
# =====================================================

elif page == "Model Evaluation":

    back_to_home()

    st.header("📈 Model Evaluation")

    st.subheader(
        "Performance Metrics"
    )

    metrics_df = pd.DataFrame(
        metrics.items(),
        columns=["Metric", "Value"]
    )

    st.dataframe(
        metrics_df,
        use_container_width=True
    )

    st.divider()

    # ---------------------------------
    # TEMPERATURE EVALUATION
    # ---------------------------------

    st.subheader(
        "Temperature Model Performance"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        f"{metrics['MAE']:.3f}"
    )

    c2.metric(
        "RMSE",
        f"{metrics['RMSE']:.3f}"
    )

    c3.metric(
        "R² Score",
        f"{metrics['R2']:.4f}"
    )

    st.divider()

    # ---------------------------------
    # RAIN EVALUATION
    # ---------------------------------

    st.subheader(
        "Rain Model Performance"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{metrics['Rain Accuracy']:.2%}"
    )

    c2.metric(
        "Precision",
        f"{metrics['Rain Precision']:.2%}"
    )

    c3.metric(
        "Recall",
        f"{metrics['Rain Recall']:.2%}"
    )

    c4.metric(
        "F1 Score",
        f"{metrics['Rain F1']:.2%}"
    )

    st.divider()

    # ---------------------------------
    # CONFUSION MATRIX
    # ---------------------------------

    st.subheader(
        "Rain Prediction Confusion Matrix"
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual No Rain",
            "Actual Rain"
        ],
        columns=[
            "Predicted No Rain",
            "Predicted Rain"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.divider()

    # ---------------------------------
    # LDA
    # ---------------------------------

    st.subheader(
        "LDA + Logistic Regression"
    )

    st.metric(
        "LDA Accuracy",
        f"{metrics['LDA Accuracy']:.2%}"
    )


# =====================================================
# 16. LIVE WEATHER DASHBOARD
# =====================================================

elif page == "Live Weather":

    back_to_home()

    st.header("🌍 Live Weather Dashboard")

    st.write(
        "Get current weather information "
        "for eight major cities of Bangladesh."
    )

    city = st.selectbox(
        "Select City",
        list(cities.keys()),
        key="live_city_page"
    )

    if st.button(
        "🔄 Refresh Weather",
        type="primary"
    ):

        get_live_weather.clear()

        try:

            current = get_live_weather(city)

            st.success(
                f"Weather loaded successfully for {city}"
            )

            st.subheader(
                f"🌤️ Weather in {city}, Bangladesh"
            )

            st.caption(
                f"Observation Time: {current['time']}"
            )

            show_weather_metrics(current)

        except Exception as e:

            st.error(
                f"Could not retrieve live weather: {e}"
            )


# =====================================================
# 17. FOOTER
# =====================================================

st.divider()

st.subheader("🌦️ WeatherWise")

st.caption(
    "Machine Learning Based Weather Prediction "
    "& Analysis System"
)

st.caption(
    "Historical Data: Open-Meteo Archive | "
    "Live Weather: Open-Meteo Forecast API"
)
