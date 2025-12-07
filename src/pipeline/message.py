# Save this file as: src/pipeline/message.py

from dataclasses import dataclass
from typing import Any
import time

@dataclass
class PipelineMessage:
    """
    A standard packet that travels through the pipeline.
    """
    data: Any              # The actual content (AudioChunk, Text, or Audio bytes)
    is_final: bool         # Is this the end of a sentence?
    timestamp: float = 0.0 # For tracking latency
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()