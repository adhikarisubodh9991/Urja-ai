import json
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

MODEL = None
SCALER_X = None
SCALER_Y = None
CONFIG = None
DATA = None
MODEL_DIR = None


def get_model_dir():
    # fixed folder naming in this version
    return BASE_DIR / 'models' / 'nepal_2025'


def load_resources():
    global MODEL, SCALER_X, SCALER_Y, CONFIG, DATA, MODEL_DIR

    MODEL_DIR = get_model_dir()
    model_path = MODEL_DIR / 'nepal_load_forecast_model.joblib'
    scaler_x_path = MODEL_DIR / 'scaler_X.pkl'
    scaler_y_path = MODEL_DIR / 'scaler_y.pkl'
    config_path = MODEL_DIR / 'config.json'
    data_path = BASE_DIR / 'data' / 'nepal_electricity_demand.csv'

    if not model_path.exists() or not data_path.exists():
        return False

    MODEL = joblib.load(str(model_path))
    SCALER_X = joblib.load(str(scaler_x_path))
    SCALER_Y = joblib.load(str(scaler_y_path))
    DATA = pd.read_csv(data_path, parse_dates=['date'])

    if config_path.exists():
        CONFIG = json.loads(config_path.read_text(encoding='utf-8'))

    return True


@app.route('/')
def home():
    return jsonify({'project': 'URJA AI'})


@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded': MODEL is not None,
        'data_loaded': DATA is not None,
        'rows': len(DATA) if DATA is not None else 0,
        'model_dir': MODEL_DIR.name if MODEL_DIR is not None else None
    })


load_resources()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
