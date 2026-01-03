import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('your_model.h5')

st.title("Handwritten Digit Recognition ✍️")
st.write("Draw a digit (0-9) below and click Predict")

# Create a drawing canvas
canvas = st.canvas(
    fill_color="white",  # Background
    stroke_width=15,
    stroke_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
)

if st.button("Predict"):
    if canvas.image_data is not None:
        # Convert canvas to grayscale 28x28
        img = Image.fromarray(canvas.image_data.astype('uint8')).convert('L')
        img = img.resize((28,28))
        img_array = np.array(img)/255.0
        img_array = img_array.reshape(1,28,28,1)

        # Predict digit
        prediction = model.predict(img_array).argmax()
        st.write(f"Predicted Digit: **{prediction}**")
