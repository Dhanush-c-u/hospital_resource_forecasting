import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load
from src.preprocessing import load_and_preprocess_multivariate

def forecast(file_path, model_path, scaler_path, target_column, label, feature_columns, n_days=7):
    # Load preprocessed full data and the fitted scaler
    data, scaler, full_feature_list = load_and_preprocess_multivariate(file_path, feature_columns)
    model = load_model(model_path, compile=False)

    input_seq = data[-30:].copy()  # last 30 days window
    preds = []

    target_index = full_feature_list.index(target_column)

    for _ in range(n_days):
        input_array = np.expand_dims(input_seq, axis=0)
        pred = model.predict(input_array, verbose=0)
        preds.append(pred[0][0])  # extract scalar value

        next_row = input_seq[-1].copy()
        next_row[target_index] = pred[0][0]  # update predicted target in next input
        input_seq = np.vstack([input_seq[1:], next_row])  # shift window

    # Inverse scale predictions
    inv_preds = scaler.inverse_transform(
        np.pad(np.array(preds).reshape(-1, 1), ((0, 0), (target_index, len(full_feature_list) - target_index - 1)), mode='constant')
    )[:, target_index]

    return pd.DataFrame(inv_preds, columns=[label])

def forecast_all():
    n_days = 7
    date_index = pd.date_range(start='2025-07-15', periods=n_days)

    df = pd.concat([
        forecast(
            'data/hospital_bed_data_enhanced.csv', 'model/icu_model.keras', 'model/icu_model_scaler.save',
            'icu_available', 'icu_forecast',
            ['icu_available', 'er_visits_rolling_mean_7', 'day_of_week', 'month', 'is_weekend', 'holiday_flag'], n_days
        ),
        forecast(
            'data/hospital_bed_data_enhanced.csv', 'model/general_model.keras', 'model/general_model_scaler.save',
            'general_available', 'general_forecast',
            ['general_available', 'er_visits_rolling_mean_7', 'day_of_week', 'month', 'is_weekend', 'holiday_flag'], n_days
        ),
        forecast(
            'data/staff_allocation_enhanced.csv', 'model/doctor_model.keras', 'model/doctor_model_scaler.save',
            'available_doctors', 'doctor_forecast',
            ['available_doctors', 'staff_absenteeism_rate', 'staff_shift_type_A', 'staff_shift_type_B', 'staff_shift_type_C', 'day_of_week', 'month', 'holiday_flag'], n_days
        ),
        forecast(
            'data/staff_allocation_enhanced.csv', 'model/nurse_model.keras', 'model/nurse_model_scaler.save',
            'available_nurses', 'nurse_forecast',
            ['available_nurses', 'staff_absenteeism_rate', 'staff_shift_type_A', 'staff_shift_type_B', 'staff_shift_type_C', 'day_of_week', 'month', 'holiday_flag'], n_days
        ),
        forecast(
            'data/ventilators_enhanced.csv', 'model/ventilator_model.keras', 'model/ventilator_model_scaler.save',
            'available_ventilators', 'ventilator_forecast',
            ['available_ventilators', 'day_of_week', 'month', 'holiday_flag'], n_days
        )
    ], axis=1)

    df.index = date_index
    return df

if __name__ == '__main__':
    print(forecast_all())
