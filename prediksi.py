import streamlit as st
import xgboost as xgb
import numpy as np
import joblib

# Memuat model XGBoost yang sudah dilatih
model = joblib.load('xgboost_model_new.pkl')

# Fungsi prediksi
def predict(ir_value, red_value):
    features = np.array([ir_value, red_value]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return float(prediction)

# Aplikasi Streamlit
st.title("Prediksi Menggunakan Model XGBoost")

ir_value = st.number_input("Masukkan nilai sensor IR:", min_value=0.0, step=0.1)
red_value = st.number_input("Masukkan nilai sensor Red:", min_value=0.0, step=0.1)
temp = st.number_input("Masukkan suhu:", min_value=-50.0, max_value=100.0, step=0.1)
bpm = st.number_input("Masukkan detak jantung:", min_value=30, max_value=200, step=1)

if st.button("Prediksi"):
    prediction = predict(ir_value, red_value)
    st.write("Hasil Prediksi:", prediction)
