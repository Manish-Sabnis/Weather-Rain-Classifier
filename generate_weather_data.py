from datetime import datetime
from meteostat import Point, Hourly
import pandas as pd


location = Point(19.0760, 72.8777)


start = datetime(2022, 1, 1)
end = datetime(2022, 12, 31)


data = Hourly(location, start, end)
df = data.fetch()


df = df.reset_index()
df['city'] = 'Mumbai'


df['rain'] = df['prcp'].apply(lambda x: 1 if x > 0 else 0)


df = df[['time', 'temp', 'dwpt', 'rhum', 'wspd', 'pres', 'rain', 'city']]


df.to_csv('weather_features.csv', index=False)

print("Saved historical weather data with rain labels")
