from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import json
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)
CORS(app)

# --- Load model dan data ---
model = joblib.load('modeljata.pkl')
le    = joblib.load('label_encoder.pkl')

with open('jata_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# --- Setup Sastrawi stemmer ---
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Kamus normalisasi slang (sama persis seperti di notebook-mu)
normalisasi = {
    'dimana'   : 'di mana',
    'kenapa'   : 'mengapa',
    'gimana'   : 'bagaimana',
    'emang'    : 'memang',
    'ngapain'  : 'untuk apa',
    'kayak'    : 'seperti',
    'banget'   : 'sangat',
    'kalo'     : 'kalau',
    'gak'      : 'tidak',
    'ga'       : 'tidak',
    'nggak'    : 'tidak',
    'udah'     : 'sudah',
    'jateng'   : 'jawa tengah',
    'mengenai' : 'tentang',
    'ceritain' : 'ceritakan',
    'jelasin'  : 'jelaskan',
    'sih'      : '',
    'deh'      : '',
    'dong'     : '',
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    for slang, baku in normalisasi.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', baku, text)
    tokens = word_tokenize(text)
    tokens_stem = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens_stem)

def ambil_respon(tag):
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return intent["respon"][0]
    return "Maaf, jawaban tidak ditemukan."

def get_response(user_input):
    clean = preprocess(user_input)
    if clean.strip() == "":
        return "Maaf, pertanyaan tidak valid."
    scores = model.decision_function([clean])[0]
    pred   = le.inverse_transform([scores.argmax()])[0]
    if scores.max() < 0.0:
        return "Maaf, saya belum memahami pertanyaan itu."
    return ambil_respon(pred)

# --- Route halaman utama ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Route API chatbot ---
@app.route('/chat', methods=['POST'])
def chat():
    data_req = request.get_json()
    user_msg = data_req.get('message', '').strip()
    if not user_msg:
        return jsonify({'response': 'Pesan tidak boleh kosong.'})
    bot_response = get_response(user_msg)
    return jsonify({'response': bot_response})

import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)