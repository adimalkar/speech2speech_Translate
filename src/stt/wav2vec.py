# Save this file as: src/stt/wav2vec.py

import torch
import numpy as np
from typing import Optional
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from src.stt.base import StreamingSTTInterface, STTResult
from src.utils.logger import logger

class Wav2VecSTT(StreamingSTTInterface):
    """
    Real-time Speech-to-Text using Wav2Vec2 (Large).
    """
    
    def __init__(self, model_name="facebook/wav2vec2-large-960h"):
        logger.info(f"Loading STT Model: {model_name}...")
        
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            self.model.eval()
            
            # Move to GPU if available (Big speedup!)
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logger.info("STT Model moved to GPU (CUDA).")
            else:
                logger.info("STT Model running on CPU.")
                
        except Exception as e:
            logger.error(f"Critical error loading STT model: {e}")
            raise

        self.sample_rate = 16000
        self.buffer = np.array([], dtype=np.float32)
        
    def process_chunk(self, audio_chunk) -> Optional[STTResult]:
        # Append new audio to our internal buffer
        self.buffer = np.append(self.buffer, audio_chunk.audio_data)
        
        # Wait for 0.5 seconds of context
        if len(self.buffer) < 16000 * 0.5: 
            return None

        return self._transcribe(is_final=False)

    def finalize(self) -> Optional[STTResult]:
        if len(self.buffer) == 0:
            return None
            
        result = self._transcribe(is_final=True)
        self.buffer = np.array([], dtype=np.float32)
        return result

    def _transcribe(self, is_final: bool) -> Optional[STTResult]:
        try:
            # --- FIX: Normalize Volume (Boost Audio) ---
            # If audio is too quiet, the AI fails. We boost it here.
            audio_data = self.buffer
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                # Normalize to -1.0 to 1.0 range (Standard volume)
                audio_data = audio_data / max_val
            # -------------------------------------------

            input_values = self.processor(
                audio_data, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            ).input_values

            # Move input to GPU if model is on GPU
            if torch.cuda.is_available():
                input_values = input_values.to("cuda")

            with torch.no_grad():
                logits = self.model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            transcription = transcription.lower()
            
            if not transcription.strip():
                return None

            return STTResult(
                text=transcription,
                confidence=1.0, 
                is_partial=not is_final,
                is_final=is_final
            )
            
        except Exception as e:
            logger.error(f"STT Error: {e}")
            return None