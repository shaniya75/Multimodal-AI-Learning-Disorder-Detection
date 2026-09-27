"""
Streamlit session-state helpers.

Centralizes the keys used to carry state between pages (HOME -> STUDENT INFO
-> SPEECH TEST -> HANDWRITING TEST -> RESULTS) so pages don't rely on
magic strings scattered throughout the codebase.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

import streamlit as st

SESSION_KEYS = [
    "student_id",
    "age",
    "grade",
    "session_id",
    "speech_audio_path",
    "speech_prediction",
    "speech_features",
    "handwriting_image_path",
    "handwriting_prediction",
]


def init_session_state() -> None:
    """Ensure all expected session-state keys exist."""
    for key in SESSION_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def set_value(key: str, value: Any) -> None:
    st.session_state[key] = value


def get_value(key: str) -> Optional[Any]:
    return st.session_state.get(key)


def reset_session() -> None:
    for key in SESSION_KEYS:
        st.session_state[key] = None


def has_student_info() -> bool:
    return bool(st.session_state.get("student_id")) and bool(st.session_state.get("session_id"))
