import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# 🔧 Fix Python path so 'src/' can be imported when running from 'dashboard/'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_summary import load_resource_status, load_er_trends
from src.forecasting import forecast_all

st.set_page_config(page_title="Hospital Resource Dashboard", layout="wide")

st.title("🏥 Hospital Resource Utilization Dashboard")

# Tabs: Dashboard | Data Entry
tab = st.sidebar.radio("📂 Select View", ["📊 Dashboard", "➕ Data Entry"])

# ===============================
# 📊 DASHBOARD TAB
# ===============================
if tab == "📊 Dashboard":
    stats = load_resource_status()

    # 🔢 Display Current Metrics
    st.markdown("### 🔢 Current Hospital Resource Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total ICU Beds", stats['icu_total'])
        st.metric("Available ICU Beds", stats['icu_available'])
        st.metric("Total General Beds", stats['general_total'])
        st.metric("Available General Beds", stats['general_available'])
    with col2:
        st.metric("Total Doctors", stats['doctor_total'])
        st.metric("Doctors Available", stats['doctors_available'])
        st.metric("Total Nurses", stats['nurse_total'])
        st.metric("Nurses Available", stats['nurses_available'])
    with col3:
        st.metric("Total Ventilators", stats['ventilator_total'])
        st.metric("Ventilators Available", stats['ventilators_available'])

    # 🚨 Emergency Room Trends
    st.markdown("### 🚨 Emergency Room Visits")
    daily, monthly = load_er_trends()
    st.subheader("🗕 Daily ER Visits")
    st.line_chart(daily)
    st.subheader("🗓 Monthly ER Visits")
    st.line_chart(monthly)

    # 📈 Forecasted Trends
    st.markdown("### 📈 Forecasted Trends (Next 7 Days)")
    try:
        forecast_df = forecast_all()
        st.subheader("🛌 Bed Forecast")
        st.line_chart(forecast_df[['icu_forecast', 'general_forecast']])
        st.subheader("👩‍⚕️ Staff Forecast")
        st.line_chart(forecast_df[['doctor_forecast', 'nurse_forecast']])
        st.subheader("💨 Ventilator Forecast")
        st.line_chart(forecast_df[['ventilator_forecast']])
        csv = forecast_df.to_csv(index=True).encode('utf-8')
        st.download_button("📂 Download Forecast Report", csv, "hospital_forecast_report.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Forecasting failed: {e}")

# ===============================
# ➕ DATA ENTRY TAB
# ===============================
elif tab == "➕ Data Entry":
    st.markdown("### 📝 Add New Daily Resource Data")

    with st.form("add_data_form", clear_on_submit=True):
        date = st.date_input("Date")

        # Bed Data
        icu_beds = st.number_input("ICU Beds", min_value=0)
        general_beds = st.number_input("General Beds", min_value=0)
        icu_available = st.number_input("Available ICU Beds", min_value=0)
        general_available = st.number_input("Available General Beds", min_value=0)
        er_visits = st.number_input("ER Visits", min_value=0)

        # Staff Data
        available_doctors = st.number_input("Available Doctors", min_value=0)
        available_nurses = st.number_input("Available Nurses", min_value=0)
        total_doctors = st.number_input("Total Doctors", min_value=0)
        total_nurses = st.number_input("Total Nurses", min_value=0)
        absenteeism = st.slider("Staff Absenteeism Rate (%)", 0, 100, 0) / 100.0
        shift_type = st.selectbox("Shift Type", ["A", "B", "C"])

        # Ventilator Data
        total_ventilators = st.number_input("Total Ventilators", min_value=0)
        available_ventilators = st.number_input("Available Ventilators", min_value=0)

        submitted = st.form_submit_button("➕ Submit Data")

        if submitted:
            try:
                day = pd.to_datetime(date)
                day_of_week = day.weekday()
                month = day.month
                is_weekend = int(day_of_week >= 5)
                holiday_flag = 0  # You can customize holiday logic later

                # Bed Data ➡ enhanced CSV
                bed_row = {
                    "date": date,
                    "icu_beds": icu_beds,
                    "general_beds": general_beds,
                    "icu_available": icu_available,
                    "general_available": general_available,
                    "er_visits": er_visits,
                    "day_of_week": day_of_week,
                    "month": month,
                    "is_weekend": is_weekend,
                    "holiday_flag": holiday_flag
                }
                bed_df = pd.read_csv("data/hospital_bed_data_enhanced.csv")
                bed_df = pd.concat([bed_df, pd.DataFrame([bed_row])], ignore_index=True)
                bed_df.to_csv("data/hospital_bed_data_enhanced.csv", index=False)

                # Staff Data ➡ enhanced CSV
                staff_row = {
                    "date": date,
                    "available_doctors": available_doctors,
                    "available_nurses": available_nurses,
                    "total_doctors": total_doctors,
                    "total_nurses": total_nurses,
                    "staff_absenteeism_rate": absenteeism,
                    "staff_shift_type_A": int(shift_type == 'A'),
                    "staff_shift_type_B": int(shift_type == 'B'),
                    "staff_shift_type_C": int(shift_type == 'C'),
                    "day_of_week": day_of_week,
                    "month": month,
                    "is_weekend": is_weekend,
                    "holiday_flag": holiday_flag
                }
                staff_df = pd.read_csv("data/staff_allocation_enhanced.csv")
                staff_df = pd.concat([staff_df, pd.DataFrame([staff_row])], ignore_index=True)
                staff_df.to_csv("data/staff_allocation_enhanced.csv", index=False)

                # Ventilator Data ➡ enhanced CSV
                vent_row = {
                    "date": date,
                    "total_ventilators": total_ventilators,
                    "available_ventilators": available_ventilators,
                    "day_of_week": day_of_week,
                    "month": month,
                    "is_weekend": is_weekend,
                    "holiday_flag": holiday_flag
                }
                vent_df = pd.read_csv("data/ventilators_enhanced.csv")
                vent_df = pd.concat([vent_df, pd.DataFrame([vent_row])], ignore_index=True)
                vent_df.to_csv("data/ventilators_enhanced.csv", index=False)

                st.success("✅ New data added successfully!")
            except Exception as e:
                st.error(f"❌ Error saving data: {e}")
