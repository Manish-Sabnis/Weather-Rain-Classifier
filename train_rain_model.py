import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


df = pd.read_csv('weather_features.csv')


features = ['temp', 'dwpt', 'rhum', 'wspd', 'pres']
X = df[features]
y = df['rain']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


joblib.dump(model, 'rain_predictor_model.pkl')
print("Model saved as rain_predictor_model.pkl")
