# 📁 evaluate_model.py

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from src.preprocessing import load_and_preprocess_multivariate

# Evaluation targets
MODELS = {
    'icu_model': {
        'path': 'data/hospital_bed_data_enhanced.csv',
        'target': 'icu_available',
        'features': ['icu_available', 'er_visits_rolling_mean_7', 'day_of_week', 'month', 'is_weekend', 'holiday_flag'],
    },
    'general_model': {
        'path': 'data/hospital_bed_data_enhanced.csv',
        'target': 'general_available',
        'features': ['general_available', 'er_visits_rolling_mean_7', 'day_of_week', 'month', 'is_weekend', 'holiday_flag'],
    },
    'doctor_model': {
        'path': 'data/staff_allocation_enhanced.csv',
        'target': 'available_doctors',
        'features': ['available_doctors', 'staff_absenteeism_rate', 'staff_shift_type_A', 'staff_shift_type_B', 'staff_shift_type_C', 'day_of_week', 'month', 'holiday_flag'],
    },
    'nurse_model': {
        'path': 'data/staff_allocation_enhanced.csv',
        'target': 'available_nurses',
        'features': ['available_nurses', 'staff_absenteeism_rate', 'staff_shift_type_A', 'staff_shift_type_B', 'staff_shift_type_C', 'day_of_week', 'month', 'holiday_flag'],
    },
    'ventilator_model': {
        'path': 'data/ventilators_enhanced.csv',
        'target': 'available_ventilators',
        'features': ['available_ventilators', 'day_of_week', 'month', 'holiday_flag'],
    }
}

for model_name, config in MODELS.items():
    print(f"\n📊 Evaluating model: model/{model_name}.keras")

    # ✅ FIXED: Only unpack 3 values
    data, scaler, full_feature_columns = load_and_preprocess_multivariate(config['path'], config['features'])

    # Prepare sequences
    X, y = [], []
    target_idx = full_feature_columns.index(config['target'])
    for i in range(30, len(data)):
        X.append(data[i-30:i])
        y.append(data[i, target_idx])
    X, y = np.array(X), np.array(y)

    # Load trained model
    model = load_model(f"model/{model_name}.keras")

    # Predict
    y_pred = model.predict(X).flatten()

    # Evaluate (scaled values)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"✅ {config['target']} — R²: {r2:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
