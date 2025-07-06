import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

model = tf.keras.models.load_model('Models/resnet50_finetuned.h5')

class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

st.title("Teeth Classification using ResNet50")
st.write("Upload an intraoral image to classify into one of the trained categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_resized = image.resize((224,224))
    img_array = np.array(img_resized)

    img_array = np.expand_dims(img_array, axis=0)

    img_preprocessed = preprocess_input(img_array)

    prediction = model.predict(img_preprocessed)
    predicted_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    st.write(f"### Prediction: **{class_names[predicted_index]}**")
    st.write(f"Confidence: {confidence:.2%}")
