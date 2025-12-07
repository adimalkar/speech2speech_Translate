# Save this file as: src/tts/base.py

from abc import ABC, abstractmethod
import numpy as np

class StreamingTTSInterface(ABC):
    """
    Abstract base class for Text-to-Speech.
    """
    
    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """
        Convert text to audio samples (numpy array).
        Returns the audio data.
        """
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """
        Return the sample rate of the generated audio.
        """
        pass