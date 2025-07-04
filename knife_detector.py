import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🔪 Knife Detection Demo")
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Run detection
    results = model.predict(image)
    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption='Detection Result', use_column_width=True)
