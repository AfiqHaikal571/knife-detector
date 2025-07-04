import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🔪 Knife Detection (Upload Image)")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model.predict(image_np)
    annotated = results[0].plot()

    st.image(annotated, caption="Detection Result", use_column_width=True)
