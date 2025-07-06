
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="edge_recycler_model_final.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# App UI
st.title("♻️ Edge Recycler Classifier")
st.write("Upload a waste image to classify it as **Organic** or **Recyclable**.")

# Upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred_class = "Recyclable" if output[0][0] > 0.5 else "Organic"
    st.success(f"🧠 Prediction: **{pred_class}**")
