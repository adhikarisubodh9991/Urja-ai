import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'nepal_electricity_demand_extended.csv'


def create_features(df):
    out = df.copy()
    out['month'] = out['date'].dt.month
    out['year'] = out['date'].dt.year
    out['quarter'] = out['date'].dt.quarter
    out['month_sin'] = np.sin(2 * np.pi * out['month'] / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['month'] / 12)
    out['season'] = out['month'].apply(lambda m: 1 if m in [6,7,8,9] else (2 if m in [10,11] else (3 if m in [12,1,2] else 4)))
    out['time_idx'] = range(len(out))

    for lag in [1,2,3,6,12]:
        out[f'lag_{lag}'] = out['demand_gwh'].shift(lag)

    out['rolling_mean_3'] = out['demand_gwh'].shift(1).rolling(3).mean()
    out['rolling_mean_6'] = out['demand_gwh'].shift(1).rolling(6).mean()
    out['rolling_mean_12'] = out['demand_gwh'].shift(1).rolling(12).mean()
    out['rolling_std_6'] = out['demand_gwh'].shift(1).rolling(6).std()
    out['rolling_std_12'] = out['demand_gwh'].shift(1).rolling(12).std()

    out['trend'] = np.arange(len(out))
    out['trend_squared'] = out['trend'] ** 2
    return out


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'missing dataset: {DATA_PATH}')

    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    feats = create_features(df).dropna().reset_index(drop=True)

    feature_cols = [
        'month', 'year', 'quarter', 'month_sin', 'month_cos', 'season', 'time_idx',
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_6', 'rolling_std_12',
        'trend', 'trend_squared'
    ]

    X = feats[feature_cols].values
    y = feats['demand_gwh'].values

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    Xs = scaler_x.fit_transform(X)
    ys = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    model = GradientBoostingRegressor(random_state=42)
    model.fit(Xs, ys)

    pred_scaled = model.predict(Xs)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)

    latest_year = int(df['date'].max().year)
    out_dir = BASE_DIR / 'models' / f'nepal_{latest_year}'
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / 'nepal_load_forecast_model.joblib')
    joblib.dump(scaler_x, out_dir / 'scaler_X.pkl')
    joblib.dump(scaler_y, out_dir / 'scaler_y.pkl')

    (out_dir / 'config.json').write_text(json.dumps({
        'feature_columns': feature_cols,
        'trained_at': datetime.now().isoformat(),
        'training_samples': int(len(feats))
    }, indent=2), encoding='utf-8')

    (out_dir / 'metrics.json').write_text(json.dumps({
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    }, indent=2), encoding='utf-8')

    print('done. saved to', out_dir)


if __name__ == '__main__':
    main()
