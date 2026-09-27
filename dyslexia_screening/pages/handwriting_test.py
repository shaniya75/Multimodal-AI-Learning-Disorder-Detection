import os

import streamlit as st

from utils.session_utils import has_student_info


def render(config: dict) -> None:
    st.title("Handwriting Test")

    if not has_student_info():
        st.warning("Please complete Student Information first.")
        if st.button("Go to Student Info"):
            st.session_state.current_page = "Student Info"
            st.rerun()
        return

    st.subheader("Upload Handwriting Sample")
    st.caption("Upload a clear photo or scan of a handwriting sample.")

    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file is not None:
        upload_dir = os.path.join("artifacts", "handwriting", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        ext = os.path.splitext(uploaded_file.name)[1] or ".png"
        save_path = os.path.join(upload_dir, f"{st.session_state.session_id}_handwriting{ext}")

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.handwriting_image_path = save_path
        st.image(uploaded_file, caption="Uploaded handwriting sample", use_column_width=True)
        st.success("Image uploaded.")

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.handwriting_image_path and st.button("Remove / Re-upload"):
            st.session_state.handwriting_image_path = None
            st.rerun()

    with col2:
        if st.button(
            "View Results", type="primary",
            disabled=st.session_state.handwriting_image_path is None,
        ):
            st.session_state.current_page = "Results"
            st.rerun()
