import streamlit as st
import xgboost as xgb
import numpy as np
import joblib
# import firebase_admin  # Dihapus untuk mencegah penggunaan Firebase
# from firebase_admin import credentials, db  # Dihapus untuk mencegah penggunaan Firebase
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import json

# Memuat model XGBoost yang sudah dilatih
model = joblib.load('xgboost_model_new.pkl')

# Mengonfigurasi Firebase (Dihapus karena Firebase tidak digunakan)
# firebase_creds = {
#     "type": st.secrets["firebase"]["type"],
#     "project_id": st.secrets["firebase"]["project_id"],
#     "private_key_id": st.secrets["firebase"]["private_key_id"],
#     "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
#     "client_email": st.secrets["firebase"]["client_email"],
#     "client_id": st.secrets["firebase"]["client_id"],
#     "auth_uri": st.secrets["firebase"]["auth_uri"],
#     "token_uri": st.secrets["firebase"]["token_uri"],
#     "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
#     "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
# }

# if not firebase_admin._apps:
#     cred = credentials.Certificate(firebase_creds)
#     firebase_admin.initialize_app(cred, {
#         'databaseURL': 'https://himedis-default-rtdb.firebaseio.com/'
#     })

# ref = db.reference('/dataSensor')  # Dihapus karena Firebase tidak digunakan

# Fungsi prediksi
def predict(ir_value, red_value):
    features = np.array([ir_value, red_value]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return float(prediction)

# Fungsi untuk memproses data baru dan memperbarui Firebase (Dihapus)
# def process_data(ir_value, red_value, temp, bpm):
#     prediction = predict(ir_value, red_value)
#     result = {
#         'irValue': ir_value,
#         'redValue': red_value,
#         'suhu': temp,
#         'bpm': bpm,
#         'prediction': prediction
#     }
#     # Menghapus atau mengomentari baris ini untuk mencegah pengiriman data ke Firebase
#     # ref.set(result)

# Kelas untuk menangani HTTP POST requests (Dihapus jika tidak diperlukan)
# class RequestHandler(BaseHTTPRequestHandler):
#     def do_POST(self):
#         content_length = int(self.headers['Content-Length'])
#         post_data = self.rfile.read(content_length)
#         data = json.loads(post_data)
#         ir_value = data.get('irValue')
#         red_value = data.get('redValue')
#         temp = data.get('suhu')
#         bpm = data.get('bpm')

#         if ir_value is None or red_value is None:
#             self.send_response(400)
#             self.end_headers()
#             self.wfile.write(b'Invalid input')
#             return

#         process_data(ir_value, red_value, temp, bpm)
#         self.send_response(200)
#         self.send_header('Content-type', 'application/json')
#         self.end_headers()
#         response = json.dumps({'prediction': prediction})
#         self.wfile.write(response.encode())

# Fungsi untuk menangani perubahan data pada Firebase (Dihapus)
# def listen_for_data_changes():
#     def listener(event):
#         if event.event_type == 'put':
#             data = event.data
#             if isinstance(data, dict):
#                 ir_value = data.get('irValue')
#                 red_value = data.get('redValue')
#                 temp = data.get('suhu')
#                 bpm = data.get('bpm')
#                 if ir_value is not None and red_value is not None:
#                     process_data(ir_value, red_value, temp, bpm)
    
#     ref.listen(listener)

# Menjalankan HTTP Server di thread terpisah (Dihapus jika tidak diperlukan)
# def run_server():
#     server_address = ('', 8000)
#     httpd = HTTPServer(server_address, RequestHandler)
#     httpd.serve_forever()

# Mulai thread untuk server HTTP (Dihapus jika tidak diperlukan)
# thread = threading.Thread(target=run_server)
# thread.daemon = True
# thread.start()

# Mulai listener Firebase (Dihapus jika tidak diperlukan)
# listen_for_data_changes()

# Aplikasi Streamlit
st.title("Prediksi Menggunakan Model XGBoost dan Firebase")

ir_value = st.number_input("Masukkan nilai sensor IR:", min_value=0.0, step=0.1)
red_value = st.number_input("Masukkan nilai sensor Red:", min_value=0.0, step=0.1)
temp = st.number_input("Masukkan suhu:", min_value=-50.0, max_value=100.0, step=0.1)
bpm = st.number_input("Masukkan detak jantung:", min_value=30, max_value=200, step=1)

if st.button("Prediksi"):
    prediction = predict(ir_value, red_value)
    result = {
        'irValue': ir_value,
        'redValue': red_value,
        'suhu': temp,
        'bpm': bpm,
        'prediction': prediction
    }
    # Menghapus atau mengomentari baris ini untuk mencegah pengiriman data ke Firebase
    # ref.set(result)  
    st.write("Hasil Prediksi:", prediction)
