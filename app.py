import json
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, jsonify, request, render_template
import joblib

app = Flask(__name__, template_folder='templates')
BASE_DIR = Path(__file__).resolve().parent

MODEL = None
SCALER_X = None
SCALER_Y = None
CONFIG = None
DATA = None
FEATURE_COLS = None
MODEL_DIR = None


def get_latest_model_dir() -> Path:
    models_root = BASE_DIR / 'models'
    if not models_root.exists():
        return models_root / 'nepal_2025'

    candidates = []
    for p in models_root.glob('nepal_*'):
        if p.is_dir() and p.name.replace('nepal_', '').isdigit():
            candidates.append((int(p.name.replace('nepal_', '')), p))

    if candidates:
        return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]

    return models_root / 'nepal_2025'


def load_resources():
    global MODEL, SCALER_X, SCALER_Y, CONFIG, DATA, FEATURE_COLS, MODEL_DIR

    MODEL_DIR = get_latest_model_dir()
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
    else:
        CONFIG = {}

    FEATURE_COLS = CONFIG.get('feature_columns', [
        'month', 'year', 'quarter', 'month_sin', 'month_cos', 'season', 'time_idx',
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_6', 'rolling_std_12',
        'trend', 'trend_squared'
    ])

    return True


def _make_x(target_date, demand_history, time_idx):
    m = target_date.month
    row = {
        'month': m,
        'year': target_date.year,
        'quarter': (m - 1) // 3 + 1,
        'month_sin': np.sin(2 * np.pi * m / 12),
        'month_cos': np.cos(2 * np.pi * m / 12),
        'season': 1 if m in [6, 7, 8, 9] else (2 if m in [10, 11] else (3 if m in [12, 1, 2] else 4)),
        'time_idx': time_idx,
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
        'trend': time_idx,
        'trend_squared': time_idx ** 2,
    }
    return np.array([[row[c] for c in FEATURE_COLS]])


def get_forecast(months_ahead=12):
    if MODEL is None:
        return {'error': 'model not loaded'}

    last_date = DATA['date'].iloc[-1]
    demand_history = list(DATA['demand_gwh'].values[-12:])
    last_idx = len(DATA) - 1

    preds, stamps = [], []

    for i in range(months_ahead):
        next_date = last_date + pd.DateOffset(months=i+1)
        x = _make_x(next_date, demand_history, last_idx + i + 1)
        x_scaled = SCALER_X.transform(x)
        pred_scaled = MODEL.predict(x_scaled)
        pred = SCALER_Y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
        preds.append(float(pred))
        stamps.append(next_date.strftime('%Y-%m'))
        demand_history.append(pred)
        demand_history.pop(0)

    return {'timestamps': stamps, 'predictions': preds, 'unit': 'GWh'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/historical')
def api_historical():
    months = request.args.get('months', 36, type=int)
    months = min(max(months, 12), len(DATA) if DATA is not None else 36)

    if DATA is None:
        return jsonify({'error': 'data not loaded'})

    df = DATA.iloc[-months:]
    return jsonify({
        'timestamps': [d.strftime('%Y-%m') for d in df['date']],
        'values': df['demand_gwh'].tolist(),
        'unit': 'GWh'
    })


@app.route('/api/forecast')
def api_forecast():
    months = request.args.get('months', 12, type=int)
    months = min(max(months, 1), 24)
    return jsonify(get_forecast(months))


@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'ok' if MODEL is not None else 'error',
        'model_loaded': MODEL is not None,
        'data_loaded': DATA is not None,
        'data_points': len(DATA) if DATA is not None else 0,
        'data_range': f"{DATA['date'].min().strftime('%Y-%m')} to {DATA['date'].max().strftime('%Y-%m')}" if DATA is not None else None,
        'model_dir': MODEL_DIR.name if MODEL_DIR is not None else None
    })


@app.route('/api/metrics')
def api_metrics():
    metrics_path = (MODEL_DIR / 'metrics.json') if MODEL_DIR is not None else None
    if metrics_path is not None and metrics_path.exists():
        return jsonify(json.loads(metrics_path.read_text(encoding='utf-8')))
    return jsonify({'error': 'metrics missing'})


@app.route('/api/annual')
def api_annual():
    if DATA is None:
        return jsonify({'error': 'data not loaded'})
    annual = DATA.groupby('fiscal_year')['demand_gwh'].sum().reset_index()
    return jsonify({
        'fiscal_years': annual['fiscal_year'].tolist(),
        'demand_twh': (annual['demand_gwh'] / 1000).tolist(),
        'unit': 'TWh'
    })


load_resources()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
