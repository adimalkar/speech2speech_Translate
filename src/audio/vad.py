# Save this file as: src/audio/vad.py
import numpy as np
import torch
from typing import Tuple, Optional
import time
from src.utils.logger import logger

class StreamingVAD:
    """
    Real-time Voice Activity Detection using Silero VAD
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        silence_duration_ms: int = 500,  # Lower wait time for snappier response
        speech_threshold: float = 0.5   # High threshold to ignore noise
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        self.silence_duration_ms = silence_duration_ms
        self.silence_frames_threshold = int(silence_duration_ms / frame_duration_ms)
        self.speech_threshold = speech_threshold
        
        # Load Silero VAD model
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self.model.eval()
            logger.info("Silero VAD model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            logger.error("Please check your internet connection or PyTorch setup.")
            raise
        
        self.get_speech_timestamps, \
        self.save_audio, \
        self.read_audio, \
        self.VADIterator, \
        self.collect_chunks = utils
        
        # State tracking
        self.silence_frames = 0
        self.is_speech_active = False # Has speech started for this utterance?
        self.frame_buffer = np.zeros(self.frame_size, dtype=np.float32)
        self.frame_count = 0
    
    def process_frame(self, audio_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Process single audio frame and return speech/silence info
        """
        start_time = time.time()
        
        # Ensure correct size
        if len(audio_frame) < self.frame_size:
            audio_frame = np.pad(
                audio_frame,
                (0, self.frame_size - len(audio_frame)),
                mode='constant'
            )
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_frame).float()
        
        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        # Determine if speech
        is_speech = speech_prob > self.speech_threshold
        
        # Update state
        if is_speech:
            self.silence_frames = 0
            self.is_speech_active = True # Mark that speech has started!
        else:
            self.silence_frames += 1
        
        self.frame_count += 1
        
        latency = (time.time() - start_time) * 1000
        if latency > 10:
            logger.debug(f"PERF: VAD processing took {latency:.1f}ms")
        
        return is_speech, speech_prob
    
    def should_end_utterance(self) -> bool:
        """
        Check if enough silence detected to mark end of utterance.
        FIX: Only return True if speech was actually active!
        """
        return self.is_speech_active and (self.silence_frames >= self.silence_frames_threshold)
    
    def reset(self):
        """Reset VAD state for next utterance"""
        self.silence_frames = 0
        self.is_speech_active = False
        self.frame_count = 0
    
    def get_state_dict(self) -> dict:
        """Get current VAD state for monitoring"""
        return {
            'silence_frames': self.silence_frames,
            'silence_frames_threshold': self.silence_frames_threshold,
            'is_speech_active': self.is_speech_active,
            'frame_count': self.frame_count
        }