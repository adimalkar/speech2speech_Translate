# Save this file as: config/config.py
import os
from dataclasses import dataclass

@dataclass
class AudioConfig:
    sample_rate: int = int(os.getenv('SAMPLE_RATE', 16000))
    # LOWER CHUNK SIZE makes the system feel snappier
    chunk_duration_ms: int = int(os.getenv('CHUNK_DURATION_MS', 200)) 
    chunk_overlap_ms: int = int(os.getenv('CHUNK_OVERLAP_MS', 0))
    vad_frame_duration_ms: int = int(os.getenv('VAD_FRAME_DURATION_MS', 32))
    
    @property
    def chunk_size(self) -> int:
        return int(self.sample_rate * self.chunk_duration_ms / 1000)
    
    @property
    def overlap_size(self) -> int:
        return int(self.sample_rate * self.chunk_overlap_ms / 1000)
    
    @property
    def stride(self) -> int:
        return self.chunk_size - self.overlap_size

# Load configurations
audio_config = AudioConfig()