# Save this file as: src/tts/mms_tts.py

import torch
import numpy as np
from transformers import VitsModel, AutoTokenizer
from src.tts.base import StreamingTTSInterface
from src.utils.logger import logger

class MMSTTS(StreamingTTSInterface):
    """
    Local, GPU-accelerated Text-to-Speech using Meta's MMS.
    Supports dynamic language loading (spa, eng, fra, deu, etc.)
    """
    
    def __init__(self, lang="spa"):
        # Dynamic Model ID construction
        # Example: 'fra' -> 'facebook/mms-tts-fra'
        model_id = f"facebook/mms-tts-{lang}"
        
        logger.info(f"Loading TTS Model: {model_id}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = VitsModel.from_pretrained(model_id)
            self.model.eval()
            
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logger.info("TTS Model moved to GPU (CUDA).")
            else:
                logger.info("TTS Model running on CPU.")
                
        except Exception as e:
            logger.error(f"Critical error loading TTS model ({model_id}): {e}")
            # Fallback to English if the specific language model doesn't exist
            if lang != "eng":
                logger.warning("Attempting fallback to English TTS...")
                try:
                    self.__init__(lang="eng")
                except:
                    raise e
            else:
                raise e

    def synthesize(self, text: str) -> np.ndarray:
        # --- FIX: Filter out bad inputs ---
        if not text or len(text.strip()) < 2:
            return np.array([])
            
        # If it's just numbers (like "50."), MMS often fails. 
        # In a real app, you'd convert "50" to "fifty".
        # For now, let's just be safe.
        if text.replace('.', '').isdigit():
            return np.array([]) 
        # ----------------------------------
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model(**inputs).waveform
            
            return output.cpu().float().numpy().flatten()
            
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            return np.array([])

    def get_sample_rate(self) -> int:
        return self.model.config.sampling_rate