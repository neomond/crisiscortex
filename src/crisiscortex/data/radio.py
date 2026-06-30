"""Radio broadcast capture and transcription pipeline.

This module handles:
1. Capturing FM/shortwave radio via RTL-SDR hardware
2. Transcribing audio using OpenAI Whisper
3. Extracting crisis-relevant signals from transcripts
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict
import warnings

import numpy as np


# Try to import Whisper - it's heavy, so we make it optional at import time
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    warnings.warn("Whisper not installed. Transcription will not work.")

# Try to import RTL-SDR - hardware is optional
try:
    from rtlsdr import RtlSdr
    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    warnings.warn("pyrtlsdr not installed. Hardware capture will not work.")


class RadioCapture:
    """Capture radio broadcasts using RTL-SDR hardware.
    
    The RTL-SDR is a $20 USB dongle that can receive:
    - FM radio (88-108 MHz)
    - Shortwave (3-30 MHz) with an upconverter
    - Air traffic, weather satellites, etc.
    
    For CrisisCortex, we focus on local FM stations that broadcast
    news in local languages (Bambara, Hausa, Arabic dialects, etc.)
    """
    
    # Common FM frequencies in crisis-prone regions
    # These are starting points - actual stations vary by location
    DEFAULT_FREQUENCIES = {
        "bamako_mali": 95.5e6,      # Example: ORTM Bamako
        "niamey_niger": 96.8e6,     # Example: Radio Niger
        "n_djamena_chad": 94.5e6,   # Example: Radiodiffusion Nationale
        "addis_ababa": 97.1e6,      # Example: Fana Broadcasting
        "mogadishu": 89.0e6,        # Example: Radio Mogadishu
    }
    
    def __init__(self, sample_rate: int = 2_048_000, ppm_error: int = 0):
        """Initialize radio capture.
        
        Args:
            sample_rate: Samples per second. 2.048 MHz is standard for FM.
            ppm_error: Frequency correction in parts-per-million.
                      Cheap dongles are off by ~50-100 ppm.
        """
        self.sample_rate = sample_rate
        self.ppm_error = ppm_error
        self.sdr = None
        
        if not RTLSDR_AVAILABLE:
            raise RuntimeError(
                "RTL-SDR not available. Install with: pip install pyrtlsdr"
            )
    
    def __enter__(self):
        """Context manager entry - opens the SDR device."""
        self.sdr = RtlSdr()
        self.sdr.sample_rate = self.sample_rate
        self.sdr.ppm = self.ppm_error
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes the SDR device."""
        if self.sdr:
            self.sdr.close()
            self.sdr = None
    
    def capture_to_file(
        self,
        frequency_hz: float,
        duration_seconds: int = 300,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Capture radio broadcast and save to WAV file.
        
        This uses rtl_fm (command-line tool) which is more reliable
        than raw pyrtlsdr for FM demodulation.
        
        Args:
            frequency_hz: Center frequency in Hz (e.g., 95.5e6 for 95.5 MHz)
            duration_seconds: How long to record
            output_path: Where to save. If None, creates temp file.
            
        Returns:
            Path to saved WAV file
        """
        if output_path is None:
            output_path = Path(tempfile.gettempdir()) / f"radio_{frequency_hz:.1f}.wav"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # rtl_fm command breakdown:
        # -f {freq}      : center frequency
        # -M wbfm        : wideband FM mode (broadcast radio)
        # -s 200000      : sample rate for FM (200 kHz)
        # -r 48000       : output sample rate (audio CD quality)
        # -p {ppm}       : frequency correction
        # -               : output to stdout
        # ffmpeg converts raw audio to proper WAV format
        
        freq_mhz = frequency_hz / 1e6
        
        cmd = (
            f"rtl_fm -f {frequency_hz:.0f} -M wbfm -s 200000 -r 48000 "
            f"-p {self.ppm_error} - | "
            f"ffmpeg -f s16le -ar 48000 -ac 1 -i - "
            f"-acodec pcm_s16le -ar 16000 -ac 1 -y {output_path}"
        )
        
        print(f"Capturing {freq_mhz:.1f} MHz for {duration_seconds}s...")
        print(f"Command: {cmd}")
        
        # Run with timeout
        try:
            subprocess.run(
                cmd,
                shell=True,
                timeout=duration_seconds + 10,  # buffer for startup
                check=True,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            print("Capture timed out - partial file saved")
        except subprocess.CalledProcessError as e:
            print(f"Capture failed: {e}")
            print(f"stderr: {e.stderr.decode() if e.stderr else 'None'}")
            raise
        
        print(f"Saved to: {output_path}")
        return Path(output_path)
    
    def list_available_frequencies(self) -> Dict[str, float]:
        """Return known frequencies for crisis monitoring regions."""
        return self.DEFAULT_FREQUENCIES.copy()


class RadioTranscriber:
    """Transcribe radio broadcasts using OpenAI Whisper.
    
    Whisper is a general-purpose speech recognition model.
    For CrisisCortex, we need to fine-tune it on local dialects
    (Bambara, Hausa, Arabic dialects) for better accuracy.
    """
    
    # Crisis-relevant keywords to flag in transcripts
    # These are starting points - the real system learns these automatically
    CRISIS_KEYWORDS = {
        "food_insecurity": [
            "famine", "hungry", "no food", "market empty", "prices doubled",
            "harvest failed", "locusts", "drought", "crops dying",
        ],
        "disease_outbreak": [
            "cholera", "fever", "sick", "hospital full", "dying", "epidemic",
            "malaria", "measles", "unknown illness",
        ],
        "conflict_escalation": [
            "militia", "attack", "fled", "displaced", "burned", "shooting",
            "road blocked", "curfew", "soldiers", "armed group",
        ],
    }
    
    def __init__(self, model_size: str = "small"):
        """Initialize Whisper model.
        
        Args:
            model_size: tiny, base, small, medium, large
                       tiny = 39M params, fastest, least accurate
                       small = 244M params, good balance
                       large = 1550M params, most accurate, slow
        """
        if not WHISPER_AVAILABLE:
            raise RuntimeError(
                "Whisper not available. Install with: pip install openai-whisper"
            )
        
        print(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        print("Model loaded")
    
    def transcribe(self, audio_path: Path) -> Dict:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to WAV, MP3, etc.
            
        Returns:
            Dictionary with:
                - text: full transcript
                - language: detected language
                - segments: timed text chunks
                - confidence: average probability
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Transcribing: {audio_path}")
        
        # Whisper transcribes and detects language automatically
        result = self.model.transcribe(
            str(audio_path),
            fp16=False,  # Use fp32 for CPU inference (more stable)
            verbose=False,
        )
        
        # Calculate average confidence
        if result.get("segments"):
            confidences = [seg.get("avg_logprob", 0) for seg in result["segments"]]
            avg_confidence = np.exp(np.mean(confidences))  # Convert logprob to probability
        else:
            avg_confidence = 0.0
        
        return {
            "text": result["text"],
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", []),
            "confidence": avg_confidence,
        }
    

    def extract_crisis_signals(self, transcript: str) -> Dict[str, List[str]]:
        """Extract crisis-relevant phrases from transcript.
        
        Matches keywords by checking if all words in the keyword
        appear anywhere in the transcript. Handles punctuation
        and natural language variations.
        
        Args:
            transcript: Full text from Whisper
            
        Returns:
            Dictionary mapping crisis type to list of matched phrases
        """
        import string
        
        # Clean transcript: lowercase, remove punctuation
        transcript_clean = transcript.lower()
        for punct in string.punctuation:
            transcript_clean = transcript_clean.replace(punct, ' ')
        
        # Split into words and remove empty strings
        words = set(w for w in transcript_clean.split() if w)
        signals = {}
        
        for crisis_type, keywords in self.CRISIS_KEYWORDS.items():
            matches = []
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Clean keyword the same way
                keyword_clean = keyword_lower
                for punct in string.punctuation:
                    keyword_clean = keyword_clean.replace(punct, ' ')
                keyword_words = [w for w in keyword_clean.split() if w]
                
                # All words in keyword must appear in transcript
                if all(kw in words for kw in keyword_words):
                    matches.append(keyword)
            if matches:
                signals[crisis_type] = matches
        
        return signals
    

    def analyze_broadcast(
        self,
        audio_path: Path,
    ) -> Dict:
        """Full pipeline: transcribe + extract crisis signals.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Complete analysis dictionary
        """
        # Step 1: Transcribe
        transcription = self.transcribe(audio_path)
        
        # Step 2: Extract signals
        signals = self.extract_crisis_signals(transcription["text"])
        
        # Step 3: Calculate simple severity score
        # More unique keywords = higher severity
        total_keywords = sum(len(v) for v in signals.values())
        severity = min(total_keywords / 5.0, 1.0)  # Cap at 1.0
        
        return {
            "transcript": transcription["text"],
            "language": transcription["language"],
            "confidence": transcription["confidence"],
            "crisis_signals": signals,
            "severity_score": severity,
            "alert": severity > 0.3,  # Threshold for flagging
        }


class MockRadioCapture:
    """Mock radio capture for testing without hardware.
    
    Use this when you don't have an RTL-SDR dongle.
    It creates synthetic audio files for development.
    """
    
    def __init__(self):
        self.frequencies = RadioCapture.DEFAULT_FREQUENCIES
    
    def capture_to_file(
        self,
        frequency_hz: float,
        duration_seconds: int = 300,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Create a silent/synthetic WAV file for testing."""
        import soundfile as sf
        
        if output_path is None:
            output_path = Path(tempfile.gettempdir()) / f"mock_radio_{frequency_hz:.1f}.wav"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate silent audio (zeros) at 16kHz
        # In real testing, you'd use a pre-recorded sample
        sample_rate = 16000
        samples = np.zeros(sample_rate * min(duration_seconds, 10))  # Max 10s for mock
        
        sf.write(output_path, samples, sample_rate)
        print(f"Created mock audio: {output_path}")
        return output_path
    
    def list_available_frequencies(self) -> Dict[str, float]:
        return self.frequencies.copy()