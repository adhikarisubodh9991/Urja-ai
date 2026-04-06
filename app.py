import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__, template_folder='templates', static_folder='static')

MODEL = None
SCALER_X = None
SCALER_Y = None
CONFIG = None
DATA = None
FEATURE_COLS = None
MODEL_DIR = None

BASE_DIR = Path(__file__).resolve().parent


def get_latest_model_dir() -> Path:
    models_root = BASE_DIR / 'models'
    if not models_root.exists():
        return models_root / 'nepal_2025'

    candidates = []
    for p in models_root.glob('nepal_*'):
        if not p.is_dir():
            continue
        suffix = p.name.replace('nepal_', '')
        if suffix.isdigit():
            candidates.append((int(suffix), p))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    return models_root / 'nepal_2025'


def load_resources():
    global MODEL, SCALER_X, SCALER_Y, CONFIG, DATA, FEATURE_COLS, MODEL_DIR

    MODEL_DIR = get_latest_model_dir()
    model_path = MODEL_DIR / 'nepal_load_forecast_model.joblib'
    scaler_X_path = MODEL_DIR / 'scaler_X.pkl'
    scaler_y_path = MODEL_DIR / 'scaler_y.pkl'
    config_path = MODEL_DIR / 'config.json'
    data_path = BASE_DIR / 'data' / 'nepal_electricity_demand.csv'

    if not model_path.exists():
        print('Model not found:', model_path)
        return False

    MODEL = joblib.load(str(model_path))
    SCALER_X = joblib.load(str(scaler_X_path))
    SCALER_Y = joblib.load(str(scaler_y_path))

    if config_path.exists():
        CONFIG = json.loads(config_path.read_text(encoding='utf-8'))
    else:
        CONFIG = {}

    FEATURE_COLS = CONFIG.get('feature_columns', [
        'month', 'year', 'quarter', 'month_sin', 'month_cos', 'season', 'time_idx',
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_6', 'rolling_std_12',
        'trend', 'trend_squared'
    ])

    DATA = pd.read_csv(data_path, parse_dates=['date'])
    return True


def create_features_for_date(target_date, demand_history, last_time_idx):
    month = target_date.month

    features = {
        'month': month,
        'year': target_date.year,
        'quarter': (month - 1) // 3 + 1,
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'season': 1 if month in [6, 7, 8, 9] else (2 if month in [10, 11] else (3 if month in [12, 1, 2] else 4)),
        'time_idx': last_time_idx + 1,
        'lag_1': demand_history[-1],
        'lag_2': demand_history[-2],
        'lag_3': demand_history[-3],
        'lag_6': demand_history[-6],
        'lag_12': demand_history[-12] if len(demand_history) >= 12 else demand_history[0],
        'rolling_mean_3': np.mean(demand_history[-3:]),
        'rolling_mean_6': np.mean(demand_history[-6:]),
        'rolling_mean_12': np.mean(demand_history[-12:]),
        'rolling_std_6': np.std(demand_history[-6:]),
        'rolling_std_12': np.std(demand_history[-12:]),
        'trend': last_time_idx,
        'trend_squared': last_time_idx ** 2,
    }

    return np.array([[features[col] for col in FEATURE_COLS]])


def get_forecast(months_ahead=12):
    if MODEL is None:
        return {'error': 'Model not loaded'}

    last_date = DATA['date'].iloc[-1]
    demand_history = list(DATA['demand_gwh'].values[-12:])
    last_time_idx = len(DATA) - 1

    predictions = []
    timestamps = []

    for i in range(months_ahead):
        next_date = last_date + pd.DateOffset(months=i + 1)

        X_new = create_features_for_date(next_date, demand_history, last_time_idx + i)
        X_new_scaled = SCALER_X.transform(X_new)

        pred_scaled = MODEL.predict(X_new_scaled)
        pred_actual = SCALER_Y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]

        predictions.append(float(pred_actual))
        timestamps.append(next_date.strftime('%Y-%m'))

        demand_history.append(pred_actual)
        demand_history.pop(0)

    return {
        'timestamps': timestamps,
        'predictions': predictions,
        'unit': 'GWh'
    }


def get_historical_data(months=36):
    if DATA is None:
        return {'error': 'Data not loaded'}

    months = min(months, len(DATA))
    recent = DATA.iloc[-months:]

    return {
        'timestamps': [d.strftime('%Y-%m') for d in recent['date']],
        'values': recent['demand_gwh'].tolist(),
        'unit': 'GWh'
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/forecast', methods=['GET', 'POST'])
def api_forecast():
    if request.method == 'POST':
        payload = request.get_json() or {}
        months = payload.get('months', 12)
    else:
        months = request.args.get('months', 12, type=int)

    months = min(max(months, 1), 24)
    return jsonify(get_forecast(months))


@app.route('/api/historical', methods=['GET'])
def api_historical():
    months = request.args.get('months', 36, type=int)
    months = min(max(months, 12), len(DATA) if DATA is not None else 108)
    return jsonify(get_historical_data(months))


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    metrics_path = (MODEL_DIR / 'metrics.json') if MODEL_DIR is not None else None
    if metrics_path is not None and metrics_path.exists():
        return jsonify(json.loads(metrics_path.read_text(encoding='utf-8')))
    return jsonify({'error': 'Metrics not found'})


@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'status': 'ok' if MODEL is not None else 'error',
        'model_loaded': MODEL is not None,
        'data_loaded': DATA is not None,
        'data_points': len(DATA) if DATA is not None else 0,
        'data_range': f"{DATA['date'].min().strftime('%Y-%m')} to {DATA['date'].max().strftime('%Y-%m')}" if DATA is not None else None
    })


@app.route('/api/annual', methods=['GET'])
def api_annual():
    if DATA is None:
        return jsonify({'error': 'Data not loaded'})

    annual = DATA.groupby('fiscal_year')['demand_gwh'].sum().reset_index()
    return jsonify({
        'fiscal_years': annual['fiscal_year'].tolist(),
        'demand_twh': (annual['demand_gwh'] / 1000).tolist(),
        'unit': 'TWh'
    })


load_resources()

if __name__ == '__main__':
    if load_resources():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        sys.exit(1)
