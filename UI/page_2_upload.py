import time
import streamlit as st
from utils.model_utils import classifier

# 引入样式文件
def local_css(file_name):
    with open(f"assets/{file_name}") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

def show_upload_page():
    st.title("📂 Enter Folder Path for Analysis")
    st.markdown(
        "<hr style='border: 1px solid #dee2e6;'>", unsafe_allow_html=True
    )

    # 确保文件路径可以在 session_state 中被持久化
    if "folder_path" not in st.session_state:
        st.session_state.folder_path = ""

    # 文件夹路径输入控件
    folder_path = st.text_input(
        "Enter the path to a folder containing images...",
        value=st.session_state.folder_path,
        key="folder_path_input"
    )

    # 将输入的文件夹路径存入 session_state
    if folder_path:
        st.session_state.folder_path = folder_path

    # 检查文件夹路径是否有效并显示 "Analyze Folder" 按钮
    if st.session_state.folder_path:
        st.success("Folder path entered successfully!")

        if st.button("Analyze Folder", key="analyze_folder"):
            with st.spinner("Analyzing images in the folder... Please wait."):
                time.sleep(2)  # 模拟延迟
                print(st.session_state.folder_path)  
                predictions = classifier.process_folder(st.session_state.folder_path)
                
                # 保存预测结果和页面状态
                st.session_state.predictions = predictions
                st.session_state.page = "results"  # 设置跳转到结果页面
                st.rerun()
