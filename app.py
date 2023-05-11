import streamlit as st
import detection_server
import requests
import json

# Streamlit app header and description
st.title("YOLOv5 Object Detection")
st.write("Upload an image and perform object detection using YOLOv5.")

# File uploader
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Send image to detection endpoint
    files = {'file': uploaded_image.getvalue()}
    response = requests.post('http://localhost:8501/detect', files=files)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response
        results = json.loads(response.text)

        # Display the detected objects
        for obj in results['objects']:
            st.write("Class:", obj['class'])
            st.write("Confidence:", obj['confidence'])
            st.image(obj['image'], use_column_width=True)
    else:
        st.error("Error occurred during object detection.")
