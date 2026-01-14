import streamlit as st
import requests
from PIL import Image
import io

# Setup the page
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")
st.title("🧠 Brain Tumor Segmentation AI")
st.write("Upload an MRI scan to detect tumors using our U-Net Model.")

# File Uploader
uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "png", "tif"])

if uploaded_file is not None:
    # 1. Show the user their uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # 2. Add a "Predict" button
    if st.button("Analyze Scan"):
        with st.spinner("Processing..."):
            # Send the image to your FastAPI Backend
            files = {"file": uploaded_file.getvalue()}
            response = requests.post("http://127.0.0.1:8000/predict", files=files)

            if response.status_code == 200:
                # Display the result
                result_image = Image.open(io.BytesIO(response.content))
                st.success("Analysis Complete!")
                st.image(result_image, caption="AI Prediction (Red Area = Tumor)", use_container_width=True)
            else:
                st.error("Error connecting to the API.")