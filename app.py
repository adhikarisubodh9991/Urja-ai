from pathlib import Path
import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'nepal_electricity_demand.csv'

DATA = None


def load_data():
    global DATA
    if not DATA_PATH.exists():
        print('data file missing:', DATA_PATH)
        return False

    DATA = pd.read_csv(DATA_PATH, parse_dates=['date'])
    DATA.sort_values('date', inplace=True)
    DATA.reset_index(drop=True, inplace=True)
    return True


@app.route('/')
def home():
    return jsonify({'project': 'URJA AI'})


@app.route('/api/status')
def status():
    return jsonify({
        'data_loaded': DATA is not None,
        'rows': int(len(DATA)) if DATA is not None else 0,
        'range': f"{DATA['date'].min().strftime('%Y-%m')} to {DATA['date'].max().strftime('%Y-%m')}" if DATA is not None else None
    })


load_data()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
