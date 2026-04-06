import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'nepal_electricity_demand_extended.csv'


def create_features(df):
    df = df.copy()

    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['season'] = df['month'].apply(lambda m: 1 if m in [6, 7, 8, 9] else (2 if m in [10, 11] else (3 if m in [12, 1, 2] else 4)))

    df['time_idx'] = range(len(df))

    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df['demand_gwh'].shift(lag)

    df['rolling_mean_3'] = df['demand_gwh'].shift(1).rolling(window=3).mean()
    df['rolling_mean_6'] = df['demand_gwh'].shift(1).rolling(window=6).mean()
    df['rolling_mean_12'] = df['demand_gwh'].shift(1).rolling(window=12).mean()
    df['rolling_std_6'] = df['demand_gwh'].shift(1).rolling(window=6).std()
    df['rolling_std_12'] = df['demand_gwh'].shift(1).rolling(window=12).std()

    df['trend'] = np.arange(len(df))
    df['trend_squared'] = df['trend'] ** 2
    return df


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    feats = create_features(df)

    feature_cols = [
        'month', 'year', 'quarter', 'month_sin', 'month_cos', 'season', 'time_idx',
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_6', 'rolling_std_12',
        'trend', 'trend_squared'
    ]

    feats = feats.dropna().reset_index(drop=True)
    X = feats[feature_cols].values
    y = feats['demand_gwh'].values

    sx = StandardScaler()
    sy = StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y.reshape(-1, 1)).ravel()

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )

    test_size = 12
    X_train, X_test = Xs[:-test_size], Xs[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    ys_train = ys[:-test_size]

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, ys_train, cv=tscv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)

    model.fit(X_train, ys_train)

    y_test_pred_scaled = model.predict(X_test)
    y_test_pred = sy.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)
    mape = np.mean(np.abs((y_test - y_test_pred) / y_test))

    model.fit(Xs, ys)

    latest_year = int(df['date'].max().year)
    out_dir = BASE_DIR / 'models' / f'nepal_{latest_year}'
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / 'nepal_load_forecast_model.joblib')
    joblib.dump(sx, out_dir / 'scaler_X.pkl')
    joblib.dump(sy, out_dir / 'scaler_y.pkl')

    (out_dir / 'config.json').write_text(json.dumps({
        'model_type': 'GradientBoostingRegressor',
        'feature_columns': feature_cols,
        'training_samples': int(len(feats)),
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }, indent=2), encoding='utf-8')

    (out_dir / 'metrics.json').write_text(json.dumps({
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'cv_rmse_mean': float(cv_rmse.mean()),
        'cv_rmse_std': float(cv_rmse.std()),
        'test_samples': int(test_size)
    }, indent=2), encoding='utf-8')

    print('saved:', out_dir)


if __name__ == '__main__':
    main()
