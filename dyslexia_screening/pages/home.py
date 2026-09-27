import streamlit as st


def render(config: dict) -> None:
    st.title(config["app"]["title"])
    st.subheader("Development Prototype — Speech and Handwriting Analysis")

    st.info(config["app"]["disclaimer"])

    st.markdown(
        """
This prototype currently contains two independent modules:

1. **English oral-reading / speech module** — built on the MPS dataset,
   using Wav2Vec 2.0 for feature extraction and a classical classifier for
   a reading-performance outcome drawn from the dataset's own annotations.
2. **Handwriting module** — built on a real handwriting dataset, using a
   ResNet18 image classifier.

The two modules are evaluated and displayed **separately**. There is no
combined score, no behavioral module, and no dyslexia diagnosis at this
stage of development.
        """
    )

    if st.button("Start Screening", type="primary"):
        st.session_state.current_page = "Student Info"
        st.rerun()
