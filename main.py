import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# TensorFlow Model Prediction
def model_prediction(test_image, threshold=0.75):
    try:
        model = tf.keras.models.load_model("trained_plant_disease_model.keras")

        # Open the image using PIL from the BytesIO object
        original_image = Image.open(test_image)
        if not is_leaf_image(original_image):
            return -2  # Indicating the image does not resemble a leaf

        # Preprocess the image for the model
        image = original_image.resize((128, 128))  # Resize directly with PIL
        input_arr = np.array(image) / 255.0  # Normalize pixel values
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        # Model prediction
        predictions = model.predict(input_arr)
        confidence = np.max(predictions)
        if confidence < threshold:
            return -1
        return np.argmax(predictions)
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        return None


# Leaf Detection Function
def is_leaf_image(image):
    """Detect if the uploaded image likely contains a leaf."""
    try:
        hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv_image, (25, 40, 40), (90, 255, 255))  # Detect green areas
        green_percentage = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
        return green_percentage > 0.1  # Adjust threshold based on your dataset
    except Exception as e:
        logger.error(f"Error during leaf detection: {e}")
        return False


# Disease Segmentation with High Accuracy
def segment_leaf_disease_accurately(image):
    """
    Detects diseased regions within a leaf boundary and highlights them directly on the uploaded image.
    """
    try:
        cv_image = np.array(image)
        if cv_image.shape[-1] == 4:  # Handle alpha channel
            cv_image = cv_image[:, :, :3]

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
        saturation_channel = hsv_image[:, :, 1]

        # Perform adaptive thresholding
        _, binary = cv2.threshold(saturation_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Detect leaf contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(binary)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        diseased_mask = cv2.inRange(gray, 120, 180)
        diseased_mask = cv2.bitwise_and(diseased_mask, diseased_mask, mask=mask)

        diseased_contours, _ = cv2.findContours(diseased_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in diseased_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return Image.fromarray(cv_image)
    except Exception as e:
        logger.error(f"Error during disease segmentation: {e}")
        return image


# Streamlit App
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    Upload an image of a plant, and our system will analyze it to detect any signs of diseases.
    Together, let's protect our crops and ensure a healthier harvest!
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### Dataset Information
    This dataset consists of about 87,000 RGB images of healthy and diseased crop leaves categorized into 38 different classes.
    - Training set: 70,295 images
    - Validation set: 17,572 images
    - Test set: 33 images

    The dataset is designed to help identify plant diseases efficiently.
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image:
        # Display the uploaded image
        image = Image.open(test_image)
        st.image(image, caption="Uploaded Image")

        if st.button("Predict Disease"):
            result_index = model_prediction(test_image)

            if result_index == -2:
                st.error("Error: The uploaded image does not appear to contain a leaf. Please upload a valid plant image.")
            elif result_index == -1:
                st.error("Error: The uploaded leaf image is not recognized. Please try a different image.")
            elif result_index is None:
                st.error("Error: Model prediction failed. Check logs for details.")
            else:
                class_name = [
                    "Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy",
                    "Blueberry__healthy", "Cherry_Powdery_mildew", "Cherry_healthy",
                    "Corn_Cercospora_leaf_spot", "Corn_Common_rust", "Corn_Northern_Leaf_Blight",
                    "Corn_healthy", "Grape_Black_rot", "Grape_Esca", "Grape_Leaf_blight",
                    "Grape_healthy", "Orange_Haunglongbing", "Peach_Bacterial_spot",
                    "Peach_healthy", "Pepper_Bacterial_spot", "Pepper_healthy",
                    "Potato_Early_blight", "Potato_Late_blight", "Potato_healthy",
                    "Raspberry_healthy", "Soybean_healthy", "Squash_Powdery_mildew",
                    "Strawberry_Leaf_scorch", "Strawberry_healthy", "Tomato_Bacterial_spot",
                    "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
                    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites", "Tomato_Target_Spot",
                    "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "Tomato_healthy"
                ]
                st.success(f"Prediction: {class_name[result_index]}")

        if st.button("Highlight Diseased Areas"):
            diseased_image = segment_leaf_disease_accurately(image.copy())
            st.image(diseased_image, caption="Image with Diseased Areas Highlighted")
