import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# TensorFlow Model Prediction
def model_prediction(image_path, threshold=0.75):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")

    # Preprocess the image for the model
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
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


# Disease Segmentation
def segment_leaf_disease(image):
    """
    Detects diseased regions within a leaf boundary and highlights them directly on the uploaded image.
    """
    # Convert PIL Image to OpenCV format
    cv_image = np.array(image)
    if cv_image.shape[-1] == 4:  # Handle alpha channel
        cv_image = cv_image[:, :, :3]

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # Perform adaptive thresholding
    _, binary = cv2.threshold(hsv_image[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect leaf contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the leaf
    mask = np.zeros_like(binary)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Detect potential diseased areas
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    diseased_mask = cv2.inRange(gray, 120, 180)
    diseased_mask = cv2.bitwise_and(diseased_mask, diseased_mask, mask=mask)

    # Highlight diseased areas
    for contour in cv2.findContours(diseased_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

    return Image.fromarray(cv_image)


# Sidebar
st.sidebar.title("Plant Disease Recognition System")
app_mode = st.sidebar.selectbox("Choose an Option", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("Welcome to the Plant Disease Recognition System 🌿🔍")
    st.image("home_page.jpeg", caption="Plant Disease Recognition")
    st.markdown("""
    This application helps identify plant diseases from images of leaves.
    - Upload an image of a plant leaf.
    - Our system will analyze the image and provide predictions for any detected diseases.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ### Dataset
    This project uses a dataset of healthy and diseased crop leaves classified into 38 categories:
    - **Training Images**: 70,295
    - **Validation Images**: 17,572
    - **Test Images**: 33
    The data was augmented and preprocessed to improve model performance.
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if test_image:
        # Display the uploaded image
        image = Image.open(test_image)
        st.image(image, caption="Uploaded Image")

        if st.button("Predict Disease"):
            if not is_leaf_image(image):
                st.error("The uploaded image does not appear to contain a leaf. Please upload a valid plant image.")
            else:
                result_index = model_prediction(test_image)
                if result_index == -1:
                    st.warning("Unrecognized disease or insufficient confidence. Please try another image.")
                else:
                    class_names = [
                        "Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy",
                        "Blueberry_healthy", "Cherry_Powdery_mildew", "Cherry_healthy",
                        "Corn_Cercospora_leaf_spot", "Corn_Common_rust", "Corn_Northern_Leaf_Blight",
                        "Corn_healthy", "Grape_Black_rot", "Grape_Esca", "Grape_Leaf_blight",
                        "Grape_healthy", "Orange_Citrus_greening", "Peach_Bacterial_spot",
                        "Peach_healthy", "Pepper_Bacterial_spot", "Pepper_healthy", "Potato_Early_blight",
                        "Potato_Late_blight", "Potato_healthy", "Raspberry_healthy", "Soybean_healthy",
                        "Squash_Powdery_mildew", "Strawberry_Leaf_scorch", "Strawberry_healthy",
                        "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
                        "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites",
                        "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus",
                        "Tomato_healthy"
                    ]
                    st.success(f"Prediction: {class_names[result_index]}")

        if st.button("Highlight Diseased Areas"):
            highlighted_image = segment_leaf_disease(image)
            st.image(highlighted_image, caption="Diseased Areas Highlighted")
