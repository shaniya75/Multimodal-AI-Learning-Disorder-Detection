import os
import subprocess

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from inference.speech_inference import run_speech_inference
from utils.config_loader import resolve_path
from utils.session_utils import has_student_info


def _load_passage(config: dict) -> str:
    passages_dir = resolve_path(config["speech"]["passages_path"])
    passage_path = os.path.join(passages_dir, "passage_01.txt")

    if os.path.exists(passage_path):
        with open(passage_path, "r", encoding="utf-8") as f:
            return f.read()

    return "(No reading passage found. Add one under data/passages/.)"


def _save_recording(audio_bytes: bytes) -> str:
    """
    Save browser-recorded audio and convert it to a proper
    16 kHz mono PCM WAV file using the FFmpeg executable
    bundled with imageio-ffmpeg.
    """

    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError(
            "imageio-ffmpeg is not installed. "
            "Run: pip install imageio-ffmpeg"
        )

    upload_dir = os.path.join(
        "artifacts",
        "speech",
        "uploads"
    )

    os.makedirs(upload_dir, exist_ok=True)

    session_id = st.session_state.session_id

    # Temporary raw browser audio
    raw_path = os.path.join(
        upload_dir,
        f"{session_id}_raw_audio"
    )

    # Final WAV used by the ML pipeline
    wav_path = os.path.join(
        upload_dir,
        f"{session_id}_speech.wav"
    )

    # Save the original browser audio bytes
    with open(raw_path, "wb") as f:
        f.write(audio_bytes)

    # Get FFmpeg bundled with imageio-ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        raw_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        wav_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Remove temporary raw recording
    if os.path.exists(raw_path):
        os.remove(raw_path)

    if result.returncode != 0:
        raise RuntimeError(
            "Could not convert the microphone recording to WAV.\n\n"
            f"FFmpeg error:\n{result.stderr}"
        )

    if not os.path.exists(wav_path):
        raise RuntimeError(
            "FFmpeg completed but the WAV file was not created."
        )

    return wav_path


def render(config: dict) -> None:
    st.title("Speech Test")

    if not has_student_info():
        st.warning("Please complete Student Information first.")

        if st.button("Go to Student Info"):
            st.session_state.current_page = "Student Info"
            st.rerun()

        return

    st.subheader("Reading Passage")

    passage = _load_passage(config)

    st.markdown(f"> {passage}")

    st.subheader("Record Reading")

    st.caption(
        "Click Start Recording and let the student read the passage. "
        "The recording will continue until you click Stop Recording."
    )

    audio = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        just_once=True,
        use_container_width=True,
        key="speech_recorder",
    )

    if audio is not None and "bytes" in audio:
        try:
            save_path = _save_recording(audio["bytes"])

            st.session_state.speech_audio_path = save_path

            st.success("Recording saved successfully.")

            st.audio(save_path)

        except Exception as exc:
            st.error(f"Could not process the recording: {exc}")

    # ---------------------------------------------------------
    # Analyze recording
    # ---------------------------------------------------------

    if st.session_state.get("speech_audio_path"):

        st.subheader("Analyze Reading")

        if st.button(
            "Analyze Recording",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                "Processing audio and generating prediction..."
            ):
                result = run_speech_inference(
                    st.session_state.speech_audio_path
                )

            if "error" in result:
                st.error(result["error"])
                return

            prediction = result.get("prediction", {})

            st.success("Speech analysis completed.")

            st.write(
                f"**Prediction:** "
                f"{prediction.get('label', 'Unknown')}"
            )

            if prediction.get("confidence") is not None:
                st.write(
                    f"**Confidence:** "
                    f"{prediction['confidence']:.2%}"
                )

            st.write(
                f"**Audio Duration:** "
                f"{result.get('audio_duration', 0):.2f} seconds"
            )

            st.write(
                f"**Sample Rate:** "
                f"{result.get('sample_rate', 0)} Hz"
            )

            st.write(
                f"**Wav2Vec2 Features:** "
                f"{result.get('embedding_dimension', 0)}"
            )

    # ---------------------------------------------------------
    # Re-record
    # ---------------------------------------------------------

    if st.session_state.get("speech_audio_path"):

        if st.button("Remove Recording / Record Again"):
            old_path = st.session_state.speech_audio_path

            if old_path and os.path.exists(old_path):
                os.remove(old_path)

            st.session_state.speech_audio_path = None

            st.rerun()

    # ---------------------------------------------------------
    # Continue
    # ---------------------------------------------------------

    if st.session_state.get("speech_audio_path"):

        if st.button(
            "Continue",
            use_container_width=True,
        ):
            st.session_state.current_page = "Handwriting Test"
            st.rerun()