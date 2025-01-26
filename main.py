import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import cv2

# Tensorflow Model Prediction
def model_prediction(test_image, threshold=0.75):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")

    # Pre-check if the image likely contains a leaf
    original_image = Image.open(test_image)
    if not is_leaf_image(original_image):
        return -2  # Indicating the image does not resemble a leaf

    # Preprocess the image for the model
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch

    # Model prediction
    predictions = model.predict(input_arr)
    confidence = np.max(predictions)  # Get the highest confidence score

    if confidence < threshold:
        return -1  # Indicating an unrecognized image
    return np.argmax(predictions)  # Return index of the class with max confidence

# Leaf Detection Function
def is_leaf_image(image):
    """Detect if the uploaded image likely contains a leaf."""
    hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv_image, (25, 40, 40), (90, 255, 255))  # Detect green areas
    green_percentage = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
    return green_percentage > 0.1  # Adjust threshold based on your dataset

# Disease Segmentation with High Accuracy
def segment_leaf_disease_accurately(image):
    """
    Detects diseased regions within a leaf boundary and highlights them directly on the uploaded image.
    """
    # Convert PIL Image to OpenCV format
    cv_image = np.array(image)
    if cv_image.shape[-1] == 4:  # Handle alpha channel
        cv_image = cv_image[:, :, :3]

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    # Extract the saturation channel
    saturation_channel = hsv_image[:, :, 1]

    # Perform adaptive thresholding
    _, binary = cv2.threshold(saturation_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect leaf contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the leaf
    mask = np.zeros_like(binary)
    if contours:
        # Assume the largest contour is the leaf
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Detect potential diseased areas (lighter patches inside the leaf)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    diseased_mask = cv2.inRange(gray, 120, 180)  # Adjust these thresholds based on image contrast
    diseased_mask = cv2.bitwise_and(diseased_mask, diseased_mask, mask=mask)  # Restrict to leaf boundary

    # Find contours of diseased regions
    diseased_contours, _ = cv2.findContours(diseased_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Highlight the diseased areas directly on the image
    for contour in diseased_contours:
        # Get the bounding box of each diseased area
        x, y, w, h = cv2.boundingRect(contour)
        # Draw circles or rectangles around the diseased area
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

    return Image.fromarray(cv_image)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    Our mission is to help in identifying plant diseases efficiently.
    Upload an image of a plant, and our system will analyze it to detect any signs of diseases.
    Together, let's protect our crops and ensure a healthier harvest!
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

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
            else:
                class_name = [
                    "Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy",
                    "Blueberry__healthy", "Cherry(including_sour)_Powdery_mildew",
                    "Cherry_(including_sour)healthy", "Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot",
                    "Corn_(maize)Common_rust", "Corn_(maize)Northern_Leaf_Blight", "Corn(maize)_healthy",
                    "Grape_Black_rot", "Grape_Esca(Black_Measles)", "Grape_Leaf_blight(Isariopsis_Leaf_Spot)",
                    "Grape_healthy", "Orange_Haunglongbing(Citrus_greening)", "Peach__Bacterial_spot",
                    "Peach_healthy", "Pepper,_bell_Bacterial_spot", "Pepper,_bell_healthy",
                    "Potato_Early_blight", "Potato_Late_blight", "Potato_healthy",
                    "Raspberry_healthy", "Soybean_healthy", "Squash_Powdery_mildew",
                    "Strawberry_Leaf_scorch", "Strawberry_healthy", "Tomato_Bacterial_spot",
                    "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
                    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites Two-spotted_spider_mite",
                    "Tomato_Target_Spot", "Tomato_Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus",
                    "Tomato___healthy"
                ]
                st.success(f"Prediction: {class_name[result_index]}")

        if st.button("Highlight Diseased Areas"):
            diseased_image = segment_leaf_disease_accurately(image.copy())
            st.image(diseased_image, caption="Uploaded Image with Diseased Areas Highlighted")
