# Save this file as: src/audio/chunk.py

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioChunk:
    """Represents a chunk of audio data flowing through the pipeline"""

    chunk_id: int  # Unique identifier
    audio_data: np.ndarray  # PCM samples (float32, -1 to 1)
    timestamp: float  # Unix timestamp when captured
    duration_ms: float  # Duration in milliseconds
    sample_rate: int  # Sampling rate (typically 16000)

    # Metadata
    vad_confidence: float = 0.0  # 0-1, speech presence confidence
    is_speech: bool = True  # Whether chunk contains speech
    is_final: bool = False  # End of utterance marker

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
        if self.max_size <= 0:
            raise ValueError(f"max_size must be a positive integer, got {self.max_size}")
        self.data = np.zeros(self.max_size, dtype=np.float32)

    def __len__(self) -> int:
        """Return the number of valid samples currently stored in the buffer."""
        return self.max_size if self.full else self.write_idx

    @property
    def is_empty(self) -> bool:
        """Return True when the buffer holds no samples."""
        return not self.full and self.write_idx == 0

    def add(self, samples: np.ndarray) -> int:
        """Append 1D audio samples to the ring buffer, keeping the latest samples on overflow."""
        samples_arr = np.asarray(samples, dtype=np.float32)
        if samples_arr.ndim != 1:
            raise ValueError(f"Expected 1D audio array, got shape {samples_arr.shape}")

        n_samples = len(samples_arr)
        if n_samples == 0:
            return 0

        if n_samples >= self.max_size:
            self.data[:] = samples_arr[-self.max_size :]
            self.write_idx = 0
            self.full = True
            return n_samples

        space_available = self.max_size - self.write_idx

        if n_samples <= space_available:
            self.data[self.write_idx : self.write_idx + n_samples] = samples_arr
            self.write_idx += n_samples
            if self.write_idx == self.max_size:
                self.write_idx = 0
                self.full = True
        else:
            # Wrap around (circular buffer logic)
            self.data[self.write_idx :] = samples_arr[:space_available]
            remainder = n_samples - space_available
            self.data[:remainder] = samples_arr[space_available:]
            self.write_idx = remainder
            self.full = True

        return n_samples

    def get_chunk(self) -> np.ndarray:
        """Get current buffered audio in chronological order."""
        if not self.full:
            return self.data[: self.write_idx].copy()
        if self.write_idx == 0:
            return self.data.copy()
        # If full/wrapped, unroll from oldest sample (write_idx) to newest (write_idx - 1)
        return np.concatenate((self.data[self.write_idx :], self.data[: self.write_idx]))

    def peek(self, n_samples: int | None = None) -> np.ndarray:
        """Return up to the most recent n_samples in chronological order without clearing."""
        if n_samples is None:
            return self.get_chunk()
        if n_samples <= 0:
            return np.zeros(0, dtype=np.float32)
        chunk = self.get_chunk()
        if n_samples >= len(chunk):
            return chunk
        return chunk[-n_samples:].copy()

    def reset(self):
        """Clear buffer without reallocating backing storage."""
        self.data.fill(0.0)
        self.write_idx = 0
        self.full = False