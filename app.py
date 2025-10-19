import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('mnist_cnn.h5')

st.title("MNIST Digit Classifier")
st.write("Upload a handwritten digit image (28x28 or larger) and the model will predict the digit.")

uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess: resize to 28x28 and invert colors if necessary
    img_resized = ImageOps.fit(image, (28,28), Image.ANTIALIAS)
    img_arr = np.array(img_resized).astype('float32') / 255.0
    
    # If background is white and digit is dark, invert so model sees dark on white or vice versa:
    # Optionally invert: if the mean pixel > 0.5, invert (heuristic)
    if img_arr.mean() > 0.5:
        img_arr = 1.0 - img_arr
    
    # reshape for model
    img_input = img_arr.reshape(1,28,28,1)
    
    # predict
    preds = model.predict(img_input)
    pred_digit = np.argmax(preds, axis=1)[0]
    confidence = preds[0][pred_digit]
    
    st.write(f"**Predicted digit:** {pred_digit}")
    st.write(f"**Confidence:** {confidence:.2f}")
