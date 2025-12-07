# Save this file as: src/mt/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class MTResult:
    """Result from Machine Translation"""
    text: str
    source_lang: str
    target_lang: str
    is_final: bool

class StreamingMTInterface(ABC):
    """
    Abstract base class for streaming machine translation.
    """
    
    @abstractmethod
    def translate_text(self, text: str, is_final: bool = False) -> Optional[MTResult]:
        """
        Translate a piece of text.
        """
        pass