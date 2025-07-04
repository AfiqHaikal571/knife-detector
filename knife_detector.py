import streamlit as st
import cv2
from ultralytics import YOLO
import os
import numpy as np

@st.cache_resource
def load_model():
    model_path = "C:/Users/TUF GAMING/Desktop/weapon/best.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at:\n{model_path}")
        st.stop()
    return YOLO(model_path)

@st.cache_resource
def get_available_cameras():
    def probe():
        available = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.read()[0]:
                available.append(i)
            cap.release()
        return available
    return probe()

model = load_model()

st.title("🔪 Real-Time Knife Detection with YOLOv11")

run = st.checkbox('Start Camera')

camera_options = get_available_cameras()
camera_index = st.selectbox("Select Camera", options=camera_options, index=0, format_func=lambda x: f"Camera {x}")

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(camera_index)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame. Exiting...")
            break
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(annotated_frame)
    cap.release()
    cv2.destroyAllWindows()
else:
    st.write("✅ Click the checkbox above to start knife detection.")
