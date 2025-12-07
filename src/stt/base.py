# Save this file as: src/stt/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class STTResult:
    """Result from STT processing"""
    text: str
    confidence: float
    is_partial: bool
    is_final: bool
    language: str = 'en'

class StreamingSTTInterface(ABC):
    """
    Abstract base class for streaming ASR implementations.
    All STT models must follow this rulebook.
    """
    
    @abstractmethod
    def process_chunk(self, audio_chunk) -> Optional[STTResult]:
        """
        Process a chunk of audio and return text (if any).
        """
        pass
    
    @abstractmethod
    def finalize(self) -> Optional[STTResult]:
        """
        Called when speech ends to get the final result.
        """
        pass