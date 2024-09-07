import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('sign_language_model.keras')

categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
              'Del', 'Nothing', 'Space']

img_height, img_width = 64, 64

def predict_image(image):
    image_resized = cv2.resize(image, (img_height, img_width))
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    return categories[predicted_class]

def predict_video():
    cap = cv2.VideoCapture(0) 
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = predict_image(frame_rgb)
        cv2.putText(frame_rgb, f"Prediction: {prediction}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        stframe.image(frame_rgb, channels="RGB")

        if st.button("Stop Video"):
            break

    cap.release()
    
st.title("ASL Alphabet Recognition (A to Z, Nothing, Space, Del)")

st.header("Upload an Image for Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = np.array(cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = predict_image(image)
    st.write(f"Predicted Sign Language Character: **{prediction}**")

st.header("Real-time Video Prediction")
if st.button("Start Video"):
    predict_video()
