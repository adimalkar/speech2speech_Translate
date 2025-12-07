# Save this as: test_tts.py
import sounddevice as sd
from src.tts.mms_tts import MMSTTS
from src.utils.logger import logger

def main():
    logger.info("--- Testing TTS (Spanish) ---")
    
    # Initialize TTS
    tts = MMSTTS(lang="spa")
    
    text = "Hola, esta es una prueba de voz generada por inteligencia artificial."
    logger.info(f"Speaking: '{text}'")
    
    # Generate Audio
    audio_data = tts.synthesize(text)
    rate = tts.get_sample_rate()
    
    logger.info(f"Audio generated: {len(audio_data)} samples @ {rate}Hz")
    
    # Play Audio
    logger.info("Playing...")
    sd.play(audio_data, rate)
    sd.wait() # Wait until audio finishes playing
    logger.info("Done.")

if __name__ == "__main__":
    main()