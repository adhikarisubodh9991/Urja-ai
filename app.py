from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'nepal_electricity_demand.csv'

DATA = None


def load_data():
    global DATA
    if not DATA_PATH.exists():
        return False
    DATA = pd.read_csv(DATA_PATH, parse_dates=['date'])
    DATA.sort_values('date', inplace=True)
    DATA.reset_index(drop=True, inplace=True)
    return True


def get_historical(months=36):
    months = min(max(months, 1), len(DATA))
    df = DATA.iloc[-months:]
    return {
        'timestamps': [d.strftime('%Y-%m') for d in df['date']],
        'values': df['demand_gwh'].tolist(),
        'unit': 'GWh'
    }


def naive_forecast(months_ahead=12):
    # basic baseline: average of last 3 months, rolled forward
    history = list(DATA['demand_gwh'].values[-6:])
    last_date = DATA['date'].iloc[-1]

    preds = []
    stamps = []

    for i in range(months_ahead):
        next_date = last_date + pd.DateOffset(months=i+1)
        pred = float(np.mean(history[-3:]))
        preds.append(pred)
        stamps.append(next_date.strftime('%Y-%m'))
        history.append(pred)

    return {'timestamps': stamps, 'predictions': preds, 'unit': 'GWh'}


@app.route('/')
def home():
    return jsonify({'project': 'URJA AI'})


@app.route('/api/historical')
def api_historical():
    months = request.args.get('months', 36, type=int)
    return jsonify(get_historical(months))


@app.route('/api/forecast')
def api_forecast():
    months = request.args.get('months', 12, type=int)
    months = min(max(months, 1), 24)
    return jsonify(naive_forecast(months))


load_data()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
