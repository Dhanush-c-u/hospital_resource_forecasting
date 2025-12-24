# 📁 train_models.py 
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import joblib
from src.preprocessing import load_and_preprocess_multivariate

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

def build_and_train(file_path, target_column, base_features, model_name):
    print(f"\n📊 Training model for: {target_column}")

    # Load preprocessed scaled data
    scaled_data, scaler, full_feature_list = load_and_preprocess_multivariate(file_path, base_features)

    print(f"ℹ️ Total features used: {len(full_feature_list)} — {full_feature_list}")

    # Prepare sequences
    X, y = [], []
    for i in range(30, len(scaled_data)):
        X.append(scaled_data[i-30:i])
        y.append(scaled_data[i, full_feature_list.index(target_column)])
    X, y = np.array(X), np.array(y)

    # Define LSTM model
    model = Sequential([
        Input(shape=(30, X.shape[2])),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())

    # Train model
    es = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=16, validation_split=0.2, callbacks=[es], verbose=1)

    # Save model and scaler
    model.save(f"model/{model_name}.keras")
    joblib.dump(scaler, f"model/{model_name}_scaler.save")
    print(f"✅ Saved model as model/{model_name}.keras")

# 🚑 Hospital Bed Models
build_and_train(
    'data/hospital_bed_data_enhanced.csv',
    'icu_available',
    ['icu_available', 'er_visits_rolling_mean_7', 'day_of_week', 'month', 'is_weekend', 'holiday_flag'],
    'icu_model'
)

build_and_train(
    'data/hospital_bed_data_enhanced.csv',
    'general_available',
    ['general_available', 'er_visits_rolling_mean_7', 'day_of_week', 'month', 'is_weekend', 'holiday_flag'],
    'general_model'
)

# 🧑‍⚕️ Staff Models
build_and_train(
    'data/staff_allocation_enhanced.csv',
    'available_doctors',
    ['available_doctors', 'staff_absenteeism_rate', 'staff_shift_type_A', 'staff_shift_type_B', 'staff_shift_type_C', 'day_of_week', 'month', 'holiday_flag'],
    'doctor_model'
)

build_and_train(
    'data/staff_allocation_enhanced.csv',
    'available_nurses',
    ['available_nurses', 'staff_absenteeism_rate', 'staff_shift_type_A', 'staff_shift_type_B', 'staff_shift_type_C', 'day_of_week', 'month', 'holiday_flag'],
    'nurse_model'
)

# 🫁 Ventilator Model
build_and_train(
    'data/ventilators_enhanced.csv',
    'available_ventilators',
    ['available_ventilators', 'day_of_week', 'month', 'holiday_flag'],
    'ventilator_model'
)
