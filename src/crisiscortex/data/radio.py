"""Radio broadcast capture and transcription pipeline."""

from pathlib import Path

import numpy as np


def capture_radio_segment(
    frequency_hz: float,
    duration_seconds: int = 300,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Capture audio from RTL-SDR at specified frequency.
    
    Args:
        frequency_hz: Center frequency in Hz (e.g., 95.5e6 for FM)
        duration_seconds: Recording duration
        sample_rate: Audio sample rate
    
    Returns:
        Raw IQ samples as numpy array
    """
    # TODO: Implement pyrtlsdr capture
    raise NotImplementedError("Radio capture not yet implemented")


def transcribe_broadcast(audio_path: Path, model_size: str = "small") -> str:
    """Transcribe radio broadcast using Whisper.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
    
    Returns:
        Transcribed text
    """
    # TODO: Implement Whisper transcription
    raise NotImplementedError("Transcription not yet implemented")
