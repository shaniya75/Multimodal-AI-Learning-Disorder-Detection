import streamlit as st

from utils.session_utils import new_session_id
from utils.validation import validate_student_id, validate_age, validate_grade


def render(config: dict) -> None:
    st.title("Student Information")
    st.caption("Only the minimum information needed for this session is collected.")

    with st.form("student_info_form"):
        student_id = st.text_input("Student ID", value=st.session_state.get("student_id") or "")
        age = st.number_input("Age", min_value=3, max_value=18, value=st.session_state.get("age") or 7, step=1)
        grade = st.text_input("Grade / Class", value=st.session_state.get("grade") or "")
        submitted = st.form_submit_button("Continue", type="primary")

    if submitted:
        try:
            student_id = validate_student_id(student_id)
            age = validate_age(age)
            grade = validate_grade(grade)
        except ValueError as exc:
            st.error(str(exc))
            return

        st.session_state.student_id = student_id
        st.session_state.age = age
        st.session_state.grade = grade
        st.session_state.session_id = new_session_id()

        st.success(f"Session created: {st.session_state.session_id}")
        st.session_state.current_page = "Speech Test"
        st.rerun()
