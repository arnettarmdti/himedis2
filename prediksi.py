import os
import json
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import joblib

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model XGBoost yang sudah dilatih
model = joblib.load('xgboost_model3.pkl')

# Mengonfigurasi Firebase menggunakan variabel lingkungan
firebase_credentials = json.loads(os.environ.get('FIREBASE_CREDENTIALS'))
cred = credentials.Certificate(firebase_credentials)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://himedislogin-default-rtdb.firebaseio.com/'
})

# Mengakses Realtime Database
ref = db.reference('/predictions')  # Sesuaikan dengan path yang sesuai di database

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mendapatkan data dari request
        data = request.json
        sensor_value_ir = data['sensor_value_ir']
        sensor_value_red = data['sensor_value_red']

        # Mengolah data sensor untuk prediksi
        features = np.array([sensor_value_ir, sensor_value_red]).reshape(1, -1)

        # Melakukan prediksi
        prediction = model.predict(features)[0]

        # Konversi prediksi ke tipe float
        prediction = float(prediction)

        # Membuat data untuk dikirim ke Realtime Database Firebase
        result = {
            'sensor_value_ir': sensor_value_ir,
            'sensor_value_red': sensor_value_red,
            'prediction': prediction
        }

        # Menyimpan data ke Realtime Database
        ref.push(result)

        # Mengirimkan hasil prediksi kembali sebagai JSON
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

# Menjalankan aplikasi Flask di Vercel
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))