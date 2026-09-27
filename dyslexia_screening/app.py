"""
Main Streamlit entry point for the
AI-Powered Early Dyslexia Screening System.

Development Stage 1:
Speech + Handwriting pipelines
"""

from __future__ import annotations

import os
import sys

import streamlit as st


# ---------------------------------------------------------
# Make project root available to Python
# ---------------------------------------------------------

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from utils.config_loader import load_config
from utils.session_utils import init_session_state

from pages import (
    home,
    student_info,
    speech_test,
    handwriting_test,
    results,
)


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Early Dyslexia Screening",
    page_icon="📘",
    layout="centered",
)


# ---------------------------------------------------------
# Page flow
# ---------------------------------------------------------

PAGE_FLOW = [
    "Home",
    "Student Info",
    "Speech Test",
    "Handwriting Test",
    "Results",
]


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

def main():

    config = load_config()

    # Initialize session variables
    init_session_state()

    # Make sure current_page exists
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    # -----------------------------------------------------
    # Sidebar navigation
    # -----------------------------------------------------

    with st.sidebar:

        st.title(
            config["app"]["title"]
        )

        st.caption(
            "Development Prototype — "
            "Speech and Handwriting Analysis"
        )

        st.divider()

        selected = st.radio(
            "Navigate",
            PAGE_FLOW,
            index=PAGE_FLOW.index(
                st.session_state.current_page
            ),
        )

        st.session_state.current_page = selected

        st.divider()

        st.caption(
            config["app"]["disclaimer"]
        )

    # -----------------------------------------------------
    # Display selected page
    # -----------------------------------------------------

    page = st.session_state.current_page

    if page == "Home":

        home.render(config)

    elif page == "Student Info":

        student_info.render(config)

    elif page == "Speech Test":

        speech_test.render(config)

    elif page == "Handwriting Test":

        handwriting_test.render(config)

    elif page == "Results":

        results.render(config)


# ---------------------------------------------------------
# Application entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()