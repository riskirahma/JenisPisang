import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
from PIL import Image

# Asumsi: Model disimpan sebagai 'banana_classifier_efficientnetv2s.keras'
MODEL_PATH = 'versi3.keras'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['ambon', 'cavendish', 'genderuwo', 'kepok', 'tanduk']

@st.cache_resource
def load_model():
    """Memuat model EfficientNetV2S yang sudah dilatih."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file '{MODEL_PATH}' ada di direktori yang sama. Error: {e}")
        return None

def preprocess_image(img):
    """Melakukan preprocessing pada gambar sesuai dengan persyaratan model EfficientNetV2S."""
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_banana_type(model, processed_img):
    """Melakukan prediksi kelas pisang menggunakan model."""
    predictions = model.predict(processed_img)
    return predictions[0]

# --- Sidebar ---
st.sidebar.title("Proyek Klasifikasi Pisang")
st.sidebar.write(
    "Aplikasi ini menggunakan **model Deep Learning EfficientNetV2S** "
    "yang telah dilatih untuk mengklasifikasikan jenis buah pisang."
)
st.sidebar.write(
    "Dataset yang digunakan untuk melatih model terdiri dari gambar-gambar "
    "pisang dari berbagai jenis."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Kelas Pisang yang Dikenali:")
for i, class_name in enumerate(CLASS_NAMES):
    st.sidebar.write(f"- {class_name.replace('_', ' ').title()}")

# --- Main Content ---
st.title("Klasifikasi Jenis Pisang")
st.write("Unggah gambar pisang untuk mendapatkan prediksinya.")

uploaded_file = st.file_uploader("Pilih gambar pisang...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Tampilkan gambar yang diunggah
        image_to_predict = Image.open(uploaded_file)
        st.image(image_to_predict, caption="Gambar yang diunggah", use_column_width=True)
        st.write("")
        st.write("Melakukan prediksi...")

        # Muat model
        model = load_model()

        if model is not None:
            # Preprocess gambar
            processed_img = preprocess_image(image_to_predict)

            # Lakukan prediksi
            predictions = predict_banana_type(model, processed_img)

            # Dapatkan kelas dengan probabilitas tertinggi
            predicted_class_index = np.argmax(predictions)
            predicted_class_name = CLASS_NAMES[predicted_class_index].replace('_', ' ').title()
            confidence = predictions[predicted_class_index] * 100

            st.markdown("---")
            st.subheader("Hasil Prediksi:")
            st.write(f"**Jenis Pisang:** `{predicted_class_name}`")
            st.write(f"**Keyakinan (Confidence):** `{confidence:.2f}%`")

            st.write("---")
            st.subheader("Detail Probabilitas:")
            # Tampilkan probabilitas untuk semua kelas
            for i, class_name in enumerate(CLASS_NAMES):
                st.write(f"- {class_name.replace('_', ' ').title()}: `{predictions[i]*100:.2f}%`")
        else:
            st.error("Model tidak dapat dimuat. Prediksi tidak dapat dilakukan.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar atau prediksi: {e}")