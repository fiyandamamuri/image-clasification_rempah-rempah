import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# Load model yang sudah disimpan
model = load_model("./model-v2.h5", compile=False)

# Load labels dari file JSON
with open("./labels.json", "r") as f:
    class_indices = json.load(f)

# Balik mapping {label: index} menjadi {index: label}
labels = {v: k for k, v in class_indices.items()}

# Dictionary informasi tambahan rempah
with open("./rempah_info.json", "r", encoding="utf-8") as f:
    rempah_info = json.load(f)

# Fungsi untuk memproses gambar sebelum prediksi
def preprocess_image(img):
    img = img.resize((224, 224))  # Sesuaikan ukuran dengan model
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalisasi
    return img

# Streamlit UI
st.title("🔍 Klasifikasi Rempah-Rempah")
st.image("./cover-dashboard.jpg", use_container_width=True)
st.subheader("Upload gambar rempah-rempah, model akan memprediksi jenisnya!")
st.caption("Note : klasifikasi hanya terbatas pada beberapa jenis rempah saja seperti, adas, andaliman, asam jawa, asam kandis, bawang bombai, bawang merah, bawang putih, bunga lawing, cabai, cengkeh, daun jeruk, daun kemangi, daun ketumbar, daun kunyit, daun pandan, daun salam, jahe, jinten, kapulaga, kayu manis, kayu secang, kemiri, kemukus, kencur, ketumbar, kluwek, kunyit, lada, lengkuas, pala, saffron, serai, temu kunci, vanili, wijen.")


uploaded_file = st.file_uploader("Unggah gambar rempah", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    st.image(img, caption="🖼️ Gambar yang diunggah", use_container_width=True)

    # Prediksi saat tombol ditekan
    if st.button("🔍 prediksi"):
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = labels[predicted_class]

        st.success(f"🎯 Prediksi : {predicted_label}")

        # Menampilkan informasi tambahan jika tersedia
        if predicted_label in rempah_info:
            info = rempah_info[predicted_label]
            st.subheader("ℹ️ **Informasi Tambahan**")
            st.write(f"**Nama:** {info['Nama']}")
            st.write(f"**Aroma:** {info['Aroma']}")
            st.write(f"**Rasa:** {info['Rasa']}")
            st.write(f"**Kegunaan:** {info['Kegunaan']}")
            st.write(f"**Manfaat Kesehatan:** {info['Manfaat Kesehatan']}")
        else:
            st.warning("⚠️ Informasi tambahan belum tersedia untuk gambar ini.")
st.caption('Copyright © Fiyanda Mamuri - 2025')
