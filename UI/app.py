import streamlit as st

# 必须将 set_page_config 放在所有命令之前
st.set_page_config(page_title="Lumbar Spine Classification", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")

from page_1_start import show_start_page
from page_2_upload import show_upload_page
from page_3_results import show_results_page

# 加载外部CSS样式
def local_css(file_name):
    with open(f"assets/{file_name}") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 引入样式文件
local_css("style.css")

# 用于存储第一张图像
first_image = None

# 初始化 session_state
if "page" not in st.session_state:
    st.session_state.page = "start"  # 设置初始页面为 "start"

# 页面控制逻辑
if st.session_state.page == "start":
    show_start_page()
elif st.session_state.page == "upload":
    show_upload_page()
elif st.session_state.page == "results":
    show_results_page()
