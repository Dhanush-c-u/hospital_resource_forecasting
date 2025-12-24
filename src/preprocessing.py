# 📁 src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_multivariate(file_path, feature_columns):
    df = pd.read_csv(file_path, parse_dates=['date'])

    # Sort by date for temporal consistency
    df.sort_values('date', inplace=True)

    lag_days = [1, 3, 7, 14]
    rolling_windows = [3, 7]
    extra_features = []

    # Add lag and rolling stats for 'available' features only
    for col in feature_columns:
        if 'available' in col:
            for lag in lag_days:
                lag_col = f"{col}_lag{lag}"
                df[lag_col] = df[col].shift(lag)
                extra_features.append(lag_col)

            for window in rolling_windows:
                roll_mean = f"{col}_roll_mean{window}"
                roll_std = f"{col}_roll_std{window}"
                df[roll_mean] = df[col].rolling(window=window).mean()
                df[roll_std] = df[col].rolling(window=window).std()
                extra_features.extend([roll_mean, roll_std])

    df.dropna(inplace=True)

    # Combine original features and newly engineered ones
    final_features = feature_columns + extra_features

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[final_features])

    return scaled, scaler, final_features
