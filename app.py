import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# -------------------------------
# Title and Description
# -------------------------------
st.title("🌊 AI-based Flood Risk Prediction System")
st.write("This project predicts flood risk using rainfall, river level, and wind speed.")

# -------------------------------
# Dataset
# -------------------------------
data = {
    "rainfall_mm":     [5, 20, 60, 100, 150, 40, 80, 120, 30, 90, 110, 70, 10, 50, 130],
    "river_level_m":   [0.3, 0.8, 1.5, 2.5, 3.5, 1.0, 2.0, 3.0, 0.9, 2.2, 2.8, 1.8, 0.4, 1.3, 3.2],
    "wind_speed_kmph": [25, 5, 12, 8, 30, 20, 10, 15, 22, 9, 18, 28, 6, 14, 11],
    "risk_level":      [0, 0, 1, 1, 2, 1, 1, 2, 0, 2, 2, 1, 0, 1, 2]
}

df = pd.DataFrame(data)

X = df[["rainfall_mm", "river_level_m", "wind_speed_kmph"]]
y = df["risk_level"]

# -------------------------------
# Train Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=3,
    min_samples_split=5,
    max_features=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Risk Message Function
# -------------------------------
def get_risk_message(risk):
    if risk == 0:
        return "🟢 LOW RISK: Situation is safe. Stay updated with weather news."
    elif risk == 1:
        return "🟡 MEDIUM RISK: Be alert. Avoid low-lying areas and stay prepared."
    elif risk == 2:
        return "🔴 HIGH RISK: Possible flood danger. Move to safer places and follow alerts."
    else:
        return "Unknown risk level"

# -------------------------------
# User Input Section
# -------------------------------
st.subheader("Enter Environmental Conditions")

rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
river_level = st.number_input("River level above normal (meters)", min_value=0.0, step=0.1)
wind_speed = st.number_input("Wind speed (km/h)", min_value=0.0, step=1.0)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Flood Risk"):
    user_data = np.array([[rainfall, river_level, wind_speed]])
    predicted_risk = model.predict(user_data)[0]

    st.subheader("Prediction Result")
    st.write("**Predicted Risk Level:**", predicted_risk)
    st.success(get_risk_message(predicted_risk))

