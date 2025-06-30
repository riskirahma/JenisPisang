# --- 0. Import Library ---
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
import base64

# --- Konstanta ---
MODEL_PATH = 'versi3.keras'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['ambon', 'cavendish', 'genderuwo', 'kepok', 'tanduk']
BG_IMAGE_PATH = "504.jpg"  # Gambar background HOME

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# --- Preprocessing Gambar ---
def preprocess_image(img):
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_banana_type(model, processed_img):
    predictions = model.predict(processed_img)
    return predictions[0]

# --- Fungsi Background HOME ---
def set_home_background(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .home-bg {{
        background-image: url("data:image/jpg;base64,{data}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        padding: 100px 20px;
        border-radius: 20px;
        box-shadow: 0 0 30px rgba(0,0,0,0.5);
    }}
    .home-title {{
        color: #f72585;
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 6px #000;
    }}
    .home-desc {{
        color: white;
        font-size: 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 4px #000;
    }}
    .home-button {{
        background: white;
        color: black;
        padding: 10px 20px;
        border-radius: 20px;
        text-decoration: none;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- Navigasi Sidebar ---
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["HOME", "OWNER", "INFO", "POSTS", "CLASSIFY"],
        icons=["house", "person-circle", "info-circle", "chat", "cloud-upload"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5!important", "background-color": "#1e1e1e"},
            "icon": {"color": "#f72585", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"#fff"},
            "nav-link-selected": {"background-color": "#7209b7"},
        },
    )

# --- Halaman HOME ---
if selected == "HOME":
    set_home_background(BG_IMAGE_PATH)
    st.markdown("""
    <div class="home-bg">
        <h1 class="home-title" style='text-align:center;'>Welcome to Banana Classifier</h1>
        <p class="home-desc" style='text-align:center;'>Klasifikasikan jenis pisang hanya dengan satu klik!<br>Powered by EfficientNetV2S.</p>
        <div style='text-align:center;'>
            <a href='#' class='home-button'>Read More</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Halaman OWNER ---
elif selected == "OWNER":
    st.markdown("""
    <div style='text-align: left; padding-left: 30px;'>
        <h2>👤 About the Owner</h2>
        <p>Dikembangkan oleh <strong>Riski Rahmadan</strong>, pengembang AI & game enthusiast.</p>
        <p>Fokus pada teknologi vision dan aplikasi edukatif berbasis AI dan game development.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Halaman INFO ---
elif selected == "INFO":
    st.markdown("""
    <div style='text-align: left; padding-left: 30px;'>
        <h2>ℹ️ Tentang Model</h2>
        <p>Model ini menggunakan arsitektur <strong>EfficientNetV2S</strong>, dilatih pada dataset gambar pisang lokal dari lima kelas:</p>
        <ul>
            <li>Ambon</li>
            <li>Cavendish</li>
            <li>Genderuwo</li>
            <li>Kepok</li>
            <li>Tanduk</li>
        </ul>
        <p>Citra diproses ulang ke resolusi 224x224 dan dinormalisasi menggunakan metode <code>preprocess_input</code>.</p>
        <p>Model ini mencapai <strong>akurasi validasi sebesar 85%</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Halaman POSTS ---
elif selected == "POSTS":
    st.markdown("""
    <div style='text-align: left; padding-left: 30px;'>
        <h2>📫 Update & Log</h2>
        <p>🚧 Belum ada update terbaru. Tetap pantau untuk fitur-fitur baru!</p>
    </div>
    """, unsafe_allow_html=True)

# --- Halaman CLASSIFY ---
elif selected == "CLASSIFY":
    st.title("🍌 Klasifikasi Pisang")
    st.write("Unggah gambar pisang untuk melihat hasil klasifikasinya.")
    uploaded_file = st.file_uploader("Upload gambar pisang", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        model = load_model()
        if model:
            processed_img = preprocess_image(img)
            predictions = predict_banana_type(model, processed_img)

            pred_index = np.argmax(predictions)
            pred_class = CLASS_NAMES[pred_index].title()
            confidence = predictions[pred_index] * 100

            st.subheader("Hasil Prediksi")
            st.write(f"**Jenis Pisang:** {pred_class}")
            st.write(f"**Tingkat Keyakinan:** {confidence:.2f}%")

            st.markdown("---")
            st.subheader("Detail Probabilitas:")
            for i, class_name in enumerate(CLASS_NAMES):
                st.write(f"- {class_name.title()}: {predictions[i]*100:.2f}%")
