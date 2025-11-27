import logging
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Initialize Whisper model (load once)
model = None


def get_model():
    """Lazy load Whisper model"""
    global model
    if model is None:
        logger.info("🎤 Loading Whisper model (medium)...")
        model = WhisperModel("medium", device="cpu", compute_type="int8")
        logger.info("✅ Whisper model loaded")
    return model


def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio file to text using faster-whisper
    
    Args:
        file_path: Path to audio file (.mp3 or .wav)
    
    Returns:
        Transcribed text as string
    
    Raises:
        Exception: If transcription fails
    """
    try:
        logger.info(f"🎧 Starting transcription: {file_path}")
        
        # Get model
        whisper_model = get_model()
        
        # Transcribe
        segments, info = whisper_model.transcribe(
            file_path,
            beam_size=5,
            language="en",  # Change to None for auto-detect
            vad_filter=True  # Voice activity detection
        )
        
        logger.info(f"📝 Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        # Combine all segments
        transcription_parts = []
        for segment in segments:
            transcription_parts.append(segment.text)
        
        full_transcription = " ".join(transcription_parts).strip()
        
        if not full_transcription:
            raise Exception("No speech detected in audio file")
        
        logger.info(f"✅ Transcription complete: {len(full_transcription)} characters")
        
        return full_transcription
        
    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        raise Exception(f"Failed to transcribe audio: {str(e)}")