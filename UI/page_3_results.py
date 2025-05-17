import streamlit as st
import pandas as pd

# 引入样式文件
def local_css(file_name):
    with open(f"assets/{file_name}") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

def show_results_page():
    st.title("📊 Analysis Results")
    st.markdown("<hr style='border: 1px solid #dee2e6;'>", unsafe_allow_html=True)

    # Display images from each series
    if "folder_images" in st.session_state and st.session_state.folder_images:
        for series, images in st.session_state.folder_images.items():
            st.subheader(f"Series: {series}")
            columns = st.columns(len(images))  # Create columns for each image in the series
            
            for i, img in enumerate(images):
                columns[i].image(img, width=200)  # Display each image in its respective column
    else:
        st.warning("No images found. Please upload a ZIP file with DICOM images and run the prediction.")

    # Display prediction tables
    st.subheader("🩺 Predictions")
    prediction_types = ["lstm", "vit", "ensemble"]
    predictions = st.session_state.get("predictions", {})

    for pred_type in prediction_types:
        if pred_type in predictions:
            st.markdown(f"### {pred_type.upper()} Predictions")
            st.table(predictions[pred_type])  # Display each prediction table
        else:
            st.warning(f"No predictions available for {pred_type.upper()}.")

    # Button to try another upload
    if st.button("🔄 Try Another", key="try_another"):
        st.session_state.page = "upload"  # Return to the upload page
        st.rerun()
