import streamlit as st
from PIL import Image

# Set the title of the app
st.title("Image Uploader")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # You can add your image processing code here
    # For example, call your Flask API to perform object detection and text recognition
    # result = your_flask_api_call(image)
    # st.write(result)
