import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained CNN model
model = tf.keras.models.load_model("digit_model.h5")

# App title
st.title("Handwritten Digit Classification")

# Upload image
uploaded_file = st.file_uploader(
    "Upload digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open image in grayscale
    image = Image.open(uploaded_file).convert('L')

    # Resize image to 28x28
    image = image.resize((28,28))

    # Show uploaded image
    st.image(image, caption="Uploaded Image", width=150)

    # Convert image to numpy array
    img_array = np.array(image)

    # Invert colors
    img_array = 255 - img_array

    # Normalize
    img_array = img_array / 255.0

    # Remove noise using threshold
    img_array = (img_array > 0.5).astype(np.float32)
    
    # Show processed image
    st.image(img_array, caption="Processed Image", width=150)

    # Reshape for CNN model
    img_array = img_array.reshape(1,28,28,1)

    # Prediction
    prediction = model.predict(img_array)

    # Predicted digit
    predicted_digit = np.argmax(prediction)

    # Confidence score
    confidence = np.max(prediction) * 100

    # Display result
    st.success(f"Predicted Digit: {predicted_digit}")

    # Display confidence
    st.write(f"Confidence: {confidence:.2f}%")