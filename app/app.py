import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

st.title("🚗 Car Object Detection")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]

if not model_files:
    st.error("No models found. Train first.")
    st.stop()

model_choice = st.selectbox("Choose Model", model_files)
model_path = os.path.join(MODEL_DIR, model_choice)

model = load_model(model_path, compile=False)

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    h, w, _ = img.shape

    resized = cv2.resize(img, (224,224))
    input_img = preprocess_input(resized)
    input_img = np.expand_dims(input_img, axis=0)

    pred_class, pred_bbox = model.predict(input_img)

    confidence = float(pred_class[0][0])
    st.write(f"Confidence: {confidence:.2f}")

    #  strict threshold
    if confidence > 0.8:
        pred_bbox = np.clip(pred_bbox[0], 0, 1)

        xmin = int(pred_bbox[0] * w)
        ymin = int(pred_bbox[1] * h)
        xmax = int(pred_bbox[2] * w)
        ymax = int(pred_bbox[3] * h)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
        st.image(img, caption="Car Detected")
    else:
        st.image(img, caption="No Car Detected")