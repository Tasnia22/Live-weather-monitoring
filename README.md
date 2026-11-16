<img width="1280" height="632" alt="image" src="https://github.com/user-attachments/assets/296901bc-b8b6-45b1-b1bd-83d143892bcb" />

# 🌦️ Live Weather Monitoring & Prediction

A Machine Learning-based weather monitoring and prediction dashboard built with **Python, Streamlit, and Scikit-learn**. This project provides live weather updates, temperature prediction, rain prediction, weather clustering, and statistical analysis for selected cities in Bangladesh.

## 🚀 Project Overview

Live Weather Monitoring combines real-time weather data with machine learning techniques to explore weather patterns and make predictions.

The dashboard is designed to make weather data easier to understand through interactive visualizations, analytical tools, and ML-based predictions.

## ✨ Features

- **Live Weather Monitoring:** Fetch current weather information using the Open-Meteo API.
- **Temperature Prediction:** Predict the next-hour temperature using Linear Regression.
- **Rain Prediction:** Classify rain conditions using Logistic Regression.
- **Weather Clustering:** Group weather observations using K-Means clustering.
- **LDA Analysis:** Apply Linear Discriminant Analysis for dimensionality reduction and classification.
- **Statistical Analysis:** Explore weather patterns and Markov transition probabilities.
- **Model Evaluation:** Evaluate model performance using standard ML metrics.

## 🧠 Machine Learning Algorithms

| Algorithm | Application |
|---|---|
| Linear Regression | Temperature prediction |
| Logistic Regression | Rain prediction |
| K-Means Clustering | Weather pattern grouping |
| Linear Discriminant Analysis (LDA) | Dimensionality reduction and classification |

## 📊 Evaluation Metrics

The project evaluates machine learning models using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Requests
- Open-Meteo API
- Jupyter Notebook

## 📍 Supported Cities

The dashboard covers eight cities in Bangladesh:

1. Dhaka
2. Chattogram
3. Rajshahi
4. Khulna
5. Sylhet
6. Barishal
7. Rangpur
8. Mymensingh

## 📂 Project Structure

```text
Live-weather-monitoring/
│
├── notebook/
│   ├── app.py
│   ├── 01_data_exploration.ipynb
│   └── bangladesh_historical_weather.csv
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tasnia22/Live-weather-monitoring.git
```

### 2. Navigate to the project directory

```bash
cd Live-weather-monitoring
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run notebook/app.py
```

The application will open in your browser at:

`http://localhost:8501`

## 🌐 Data Source

Live weather information is collected using the [Open-Meteo API](https://open-meteo.com/).

Historical weather observations are stored in a CSV dataset and used for model training and evaluation.

## 🎯 Project Objectives

- Understand and analyze weather patterns in Bangladesh.
- Apply machine learning algorithms to real-world weather data.
- Explore temperature and rain prediction.
- Visualize weather information through an interactive dashboard.
- Compare machine learning model performance.

## 👩‍💻 Author

**Tasnia Rahman**

CSE Student | Robotics Enthusiast | Aspiring Engineer

GitHub: [@Tasnia22](https://github.com/Tasnia22)

---

⭐ If you find this project interesting, consider giving it a star!
