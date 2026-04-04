from pathlib import Path
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


def historical(months=36):
    if DATA is None:
        return {'error': 'data not loaded'}

    months = min(max(months, 1), len(DATA))
    df = DATA.iloc[-months:]
    return {
        'timestamps': [d.strftime('%Y-%m') for d in df['date']],
        'values': df['demand_gwh'].tolist(),
        'unit': 'GWh'
    }


@app.route('/')
def home():
    return jsonify({'project': 'URJA AI'})


@app.route('/api/status')
def status():
    return jsonify({'ok': DATA is not None, 'rows': len(DATA) if DATA is not None else 0})


@app.route('/api/historical')
def api_historical():
    months = request.args.get('months', 36, type=int)
    return jsonify(historical(months))


load_data()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
