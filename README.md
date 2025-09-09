# Real-Time Weather Rain Classifier

Predict rain in real-time using weather data from free APIs. This project includes a trained ML model, an interactive Streamlit dashboard, and automated workflows using GitHub Actions.

## Demo
[Live Streamlit App](https://weather-rain-classifier-fa55ivjbi3nxuzs5lurn8q.streamlit.app/)

Here is a screenshot in case the app is asleep
<img width="736" height="543" alt="image" src="https://github.com/user-attachments/assets/b8eb3bf8-a385-426d-8723-e3afd2634446" />


## Features
- Real-time weather data fetched via Meteostat API
- Binary classification: Rain or No Rain
- Trained using Random Forest on features like temperature, humidity, wind, pressure, etc.
- Automated hourly predictions using GitHub Actions
- Weekly retraining pipeline
- Streamlit dashboard for visualization

---

## Model Workflow

### Training (`train_rain_model.py`)
- Uses historical weather data
- Extracts relevant features
- Trains a `RandomForestClassifier`
- Saves the model as `model.pkl`

### Prediction (`predict_rain.py`)
- Runs every hour (via GitHub Actions)
- Fetches the most recent weather data
- Loads trained model
- Makes prediction → logs/stores result

---

## Project Structure
├── app.py # Streamlit dashboard

├── predict_rain.py # Hourly prediction script

├── train_rain_model.py # Weekly model training script

├── fetch_features.py # Fetch historical features for training

├── model.pkl # Trained model (auto-updated)

├── requirements.txt

├── .github/

│ ├── workflows/

│ │ ├── predict.yml # GitHub Action: hourly predictions

│ │ └── train.yml # GitHub Action: weekly retraining



---

## Automation with GitHub Actions

### `predict.yml`
- Runs **hourly**
- Executes `predict_rain.py`
- Makes live predictions with the latest data

### `train.yml`
- Runs **weekly (Monday @ 00:00)**
- Fetches past data, trains a new model
- Replaces the old model in the repo

---

## Setup Locally

```bash
git clone https://github.com/your-username/Weather-Rain-Classifier.git
cd Weather-Rain-Classifier
pip install -r requirements.txt
streamlit run app.py
```

---


## Credits
- Built by [Manish Sabnis](https://github.com/Manish-Sabnis)
- Inspired by [Pau Labarta Bajo](https://www.linkedin.com/in/pau-labarta-bajo-4432074b/)’s MLOps project walkthrough

