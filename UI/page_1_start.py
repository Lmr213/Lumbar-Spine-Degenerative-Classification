import streamlit as st

# 引入样式文件
def local_css(file_name):
    with open(f"assets/{file_name}") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

def show_start_page():
    st.title("🩺 Lumbar Spine Degenerative Classification")
    st.markdown(
        """
        <p style='font-size:18px; color:#6c757d;'>
        Welcome to the Lumbar Spine Analysis Tool. Start by uploading your images for detailed analysis.
        </p>
        """, 
        unsafe_allow_html=True
    )

    if st.button("Start Analysis", key="start"):
        st.session_state.page = "upload"
        st.rerun()  