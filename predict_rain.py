from datetime import datetime, timedelta
from meteostat import Point, Hourly
import pandas as pd
import joblib
from datetime import timezone

model = joblib.load('rain_predictor_model.pkl')


location = Point(19.0760, 72.8777)


end = datetime.utcnow()
start = end - timedelta(hours=1)


data = Hourly(location, start, end)
df = data.fetch()

if df.empty:
    print("No recent weather data available.")
else:
    latest = df.iloc[-1]
    features = ['temp', 'dwpt', 'rhum', 'wspd', 'pres']
    X_input = pd.DataFrame([latest[features].values], columns=features)
    prediction = model.predict(X_input)[0]

    proba = model.predict_proba(X_input)[0]

    result = "Rain" if prediction == 1 else "No Rain"
    print(f"Prediction for Mumbai at {latest.name} → {result}")
    print(f"Confidence → Rain: {proba[1]*100:.1f}%, No Rain: {proba[0]*100:.1f}%")
