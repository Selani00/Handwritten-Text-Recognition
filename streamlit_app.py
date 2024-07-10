import streamlit as st
import requests
from PIL import Image
import io

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

    # Convert the uploaded image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()

    # Send the image to the Flask API
    response = requests.post(
        'http://127.0.0.1:5000/upload',
        files={'image': ('image.jpg', image_bytes, 'image/jpeg')}
    )

    if response.status_code == 200:
        result = response.json()
        st.write('Objects detected:', result['objects'])

        if result['person_detected']:
            st.write('Person detected in the image.')
            st.write('Text detected in the image:')
            for text_result in result['text_results']:
                st.write(f"Text: {text_result['text']}, Probability: {text_result['probability']}")
        else:
            st.write('No person detected in the image.')
    else:
        st.write('Error processing image.')
