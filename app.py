GNU nano 8.5                               app.py
import streamlit as st
import pickle
import cv2
import numpy as np

# Load trained model
model = pickle.load(open("image_classifier.pkl", "rb"))

st.title(" M-6 M-1 Cat vs Dog Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert file to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", caption="Uploaded Image")

    # Preprocess image like in train.py
    img = cv2.resize(img, (100, 100))   # same size used in training
    img = img.flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(img)

    if prediction[0] == 0:
        st.success("This is a **Cat  M-1**")
    else:
        st.success("This is a **Dog  M-6**")

st.title("OpenCV Image Classifier")
st.write("Hello! Deployment test working ✅")
