import streamlit as st
from datetime import datetime, timedelta
from meteostat import Point, Hourly
import joblib
import pandas as pd

st.title("Real-Time Rain Predictor - Mumbai")

# Load model
model = joblib.load('rain_predictor_model.pkl')

# Fetch recent data
location = Point(19.0760, 72.8777)
end = datetime.utcnow()
start = end - timedelta(hours=1)
data = Hourly(location, start, end).fetch()

if data.empty:
    st.warning("No recent weather data available.")
else:
    latest = data.iloc[-1]
    features = ['temp', 'dwpt', 'rhum', 'wspd', 'pres']
    X_input = latest[features].values.reshape(1, -1)

    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    st.subheader("Latest Weather Conditions (Mumbai)")
    st.write(latest[features])

    st.subheader("Prediction")
    if prediction == 1:
        st.error(f"It may RAIN")
    else:
        st.success(f"No Rain")

    st.text(f"Confidence → Rain: {proba[1]*100:.1f}%, No Rain: {proba[0]*100:.1f}%")
