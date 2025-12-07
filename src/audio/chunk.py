# Save this file as: src/audio/chunk.py

from dataclasses import dataclass, field
import numpy as np
from typing import Optional
import time

@dataclass
class AudioChunk:
    """Represents a chunk of audio data flowing through the pipeline"""
    
    chunk_id: int                          # Unique identifier
    audio_data: np.ndarray                 # PCM samples (float32, -1 to 1)
    timestamp: float                       # Unix timestamp when captured
    duration_ms: float                     # Duration in milliseconds
    sample_rate: int                       # Sampling rate (typically 16000)
    
    # Metadata
    vad_confidence: float = 0.0            # 0-1, speech presence confidence
    is_speech: bool = True                 # Whether chunk contains speech
    is_final: bool = False                 # End of utterance marker
    
    # Tracking
    pipeline_start_time: float = field(default_factory=time.time)
    
    def get_latency_ms(self) -> float:
        """Calculate latency from capture to now"""
        return (time.time() - self.pipeline_start_time) * 1000

@dataclass
class AudioBuffer:
    """Circular buffer for handling streaming audio input"""
    
    max_size: int
    data: np.ndarray = field(init=False)
    write_idx: int = 0
    full: bool = False
    
    def __post_init__(self):
        self.data = np.zeros(self.max_size, dtype=np.float32)
    
    def add(self, samples: np.ndarray) -> int:
        """Add samples to buffer"""
        n_samples = len(samples)
        space_available = self.max_size - self.write_idx
        
        if n_samples <= space_available:
            self.data[self.write_idx:self.write_idx + n_samples] = samples
            self.write_idx += n_samples
            if self.write_idx == self.max_size:
                self.full = True
        else:
            # Wrap around (circular buffer logic)
            self.data[self.write_idx:] = samples[:space_available]
            self.data[:n_samples - space_available] = samples[space_available:]
            self.write_idx = n_samples - space_available
            self.full = True
        
        return n_samples
    
    def get_chunk(self) -> np.ndarray:
        """Get current buffered audio"""
        if not self.full:
            return self.data[:self.write_idx].copy()
        # If full/wrapped, we need to unroll it to get chronological order
        return np.concatenate((self.data[self.write_idx:], self.data[:self.write_idx]))

    def reset(self):
        """Clear buffer"""
        self.data = np.zeros(self.max_size, dtype=np.float32)
        self.write_idx = 0
        self.full = False