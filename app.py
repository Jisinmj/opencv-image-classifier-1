import streamlit as st
import cv2
import pickle
import numpy as np
from PIL import Image

# load model
model = pickle.load(open("image_classifier.pkl", "rb"))

st.title("OpenCV Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocessing
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (64,64)).flatten().reshape(1,-1)

    # prediction
    pred = model.predict(img_array)
    st.subheader("Prediction")
    st.write(f"Predicted class: **{pred[0]}**")

