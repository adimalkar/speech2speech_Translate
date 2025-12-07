import numpy as np
import torch
from typing import Optional
from faster_whisper import WhisperModel
from src.stt.base import StreamingSTTInterface, STTResult
from src.utils.logger import logger

class WhisperSTT(StreamingSTTInterface):
    def __init__(self, model_size="small.en", device="cuda"):
        logger.info(f"Loading Whisper Model: {model_size}...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type="float16")
            logger.info("Whisper Model loaded successfully.")
        except Exception as e:
            logger.error(f"Critical error loading Whisper: {e}")
            raise

        self.buffer = np.array([], dtype=np.float32)
        self.sample_rate = 16000
        
        # Minimum audio duration to avoid hallucinations but allow short words
        # (single-word utterances like "hello" are ~0.4-0.8s)
        self.min_audio_duration = 0.4
        
        # Whisper hallucinations - ONLY things that indicate no real speech
        # Don't include real words like "hello", "yes", "no" - those could be intentional!
        self.hallucinations = {
            "you", "the", "a", "i", "it", "is", "to", "in", "and",
            "um", "uh", "hmm", "ah",
            "thanks for watching", "thank you for watching",
            "subtitles by", "subtitle by", "copyright",
            "please subscribe", "like and subscribe"
        }

    def process_chunk(self, audio_chunk) -> Optional[STTResult]:
        self.buffer = np.append(self.buffer, audio_chunk.audio_data)
        return None

    def finalize(self) -> Optional[STTResult]:
        if len(self.buffer) == 0:
            return None
        
        # Calculate audio duration
        duration = len(self.buffer) / self.sample_rate
        # Force-finalize very long speech to avoid waiting for silence
        if duration > 5.0:
            logger.debug(f"Auto-finalizing long buffer ({duration:.2f}s)")
        
        # Reject audio that's too short - these cause hallucinations
        if duration < self.min_audio_duration:
            logger.debug(f"Audio too short ({duration:.2f}s < {self.min_audio_duration}s), skipping")
            self.buffer = np.array([], dtype=np.float32)
            return None
            
        audio_data = self.buffer
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Reject segments that are essentially silence (very low energy)
        rms = np.sqrt(np.mean(np.square(audio_data)))
        if rms < 1e-4:
            logger.debug(f"Audio RMS too low ({rms:.2e}), skipping as silence")
            self.buffer = np.array([], dtype=np.float32)
            return None

        try:
            segments, info = self.model.transcribe(
                audio_data, 
                beam_size=5, 
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,  # Use Whisper's built-in VAD to filter silence
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            text = " ".join([segment.text for segment in segments]).strip()
            
            self.buffer = np.array([], dtype=np.float32)
            
            if not text:
                return None

            clean_text = text.lower().strip(" .!?-")
            
            # Filter out known hallucinations (single words or short phrases)
            if clean_text in self.hallucinations:
                logger.debug(f"Filtered hallucination: '{text}'")
                return None
            
            # Allow short words (e.g., "ok", "hi", "cool"); only reject single-char
            if len(clean_text) < 1:
                logger.debug(f"Filtered empty/short text: '{text}'")
                return None
            
            return STTResult(text=text, confidence=1.0, is_partial=False, is_final=True)
            
        except Exception as e:
            logger.error(f"Whisper Error: {e}")
            self.buffer = np.array([], dtype=np.float32)
            return None