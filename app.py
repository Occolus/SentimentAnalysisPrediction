import streamlit as st
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Gojek Sentiment Dashboard")

# ========= LOAD MODEL & PREPROCESSOR =========
@st.cache_resource

@st.cache_resource
def load_logreg_and_tfidf():
    try:
        tfidf = joblib.load("tfidf_vectorizer.joblib")
        logreg = joblib.load("logistic_regression_model.joblib")
        return tfidf, logreg
    except Exception as e:
        st.warning(f"Gagal load TF-IDF / Logistic Regression: {e}")
        return None, None

tfidf_vectorizer, logreg_model = load_logreg_and_tfidf()

MAX_LEN = 100  # samakan dengan yang dipakai saat training LSTM

label_map = {0: "Negatif", 1: "Positif"}   # sesuaikan jika encoding berbeda


# ========= FUNGSI PREDIKSI =========
def predict_with_lstm(text: str):
    if lstm_model is None or tokenizer is None:
        return None, None
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(lstm_model.predict(padded)[0][0])
    label = 1 if prob >= 0.5 else 0
    return label_map[label], prob

def predict_with_logreg(text: str):
    if tfidf_vectorizer is None or logreg_model is None:
        return None, None
    X = tfidf_vectorizer.transform([text])
    label = int(logreg_model.predict(X)[0])
    prob = None
    if hasattr(logreg_model, "predict_proba"):
        prob = float(logreg_model.predict_proba(X)[0].max())
    return label_map[label], prob


# ========= UI DASHBOARD =========
st.title("Dashboard Analisis Sentimen Review Gojek")

st.markdown("Masukkan komentar pengguna, sistem akan memprediksi apakah sentimennya **positif** atau **negatif**.")

model_choice = st.radio(
    "Pilih model untuk prediksi:",
    ( "Logistic Regression TF-IDF"),
)

user_input = st.text_area(
    "Komentar pengguna:",
    placeholder="Contoh: Aplikasinya sering error dan susah login...",
    height=150,
)

if st.button("Prediksi Sentimen"):
    if not user_input.strip():
        st.error("Silakan masukkan komentar terlebih dahulu.")
    else:
        if model_choice == "LSTM (Deep Learning)":
            label, prob = predict_with_lstm(user_input)
        else:
            label, prob = predict_with_logreg(user_input)

        if label is None:
            st.error("Model belum berhasil dimuat. Periksa kembali file di folder streamlit_models.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediksi Sentimen", label)
            with col2:
                if prob is not None:
                    st.metric("Kepercayaan Model", f"{prob*100:.1f}%")
                else:
                    st.metric("Kepercayaan Model", "N/A")

            st.subheader("Detail Input")
            st.write(user_input)
