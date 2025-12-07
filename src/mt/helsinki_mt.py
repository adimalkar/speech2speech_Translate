import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import Optional
from src.mt.base import StreamingMTInterface, MTResult
from src.utils.logger import logger

class HelsinkiMT(StreamingMTInterface):
    """
    Machine Translation using Helsinki-NLP/MarianMT models.
    """
    
    def __init__(self, source_lang: str = "en", target_lang: str = "es"):
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Construct model name dynamically (e.g., Helsinki-NLP/opus-mt-en-es)
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        logger.info(f"Loading MT Model: {model_name}...")
        
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            self.model.eval()
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            logger.info(f"MT Model loaded successfully on {self.device}.")
            
        except Exception as e:
            logger.error(f"Critical error loading MT model {model_name}: {e}")
            raise

    def translate_text(self, text: str, is_final: bool = False) -> Optional[MTResult]:
        """
        Translates text from source_lang to target_lang.
        """
        if not text or not text.strip():
            return None
            
        try:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                translated_tokens = self.model.generate(**inputs)
            
            # Decode
            translated_text = self.tokenizer.batch_decode(
                translated_tokens, 
                skip_special_tokens=True
            )[0]
            
            return MTResult(
                text=translated_text,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                is_final=is_final
            )
            
        except Exception as e:
            logger.error(f"MT Error during translation: {e}")
            return None

