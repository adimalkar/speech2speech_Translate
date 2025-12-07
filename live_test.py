# Save this file as: live_test.py

import time
import sys
import numpy as np
from queue import Empty
from src.audio.capture import StreamingAudioCapture
from src.stt.wav2vec import Wav2VecSTT
from src.utils.logger import logger

def main():
    logger.info("--- LIVE SPEECH-TO-TEXT DEBUG MODE ---")
    
    logger.info("1. Loading AI Model...")
    stt = Wav2VecSTT()
    
    logger.info("2. Initializing Microphone (Device 4 @ 48000Hz)...")
    capture = StreamingAudioCapture(
        enable_vad=True,
        device_index=4,
        mic_sample_rate=48000
    )
    
    capture_thread = capture.start()
    
    logger.info("Listening...")
    logger.info("Legend: '.'=Silence, '?'=Processing Speech, [TEXT]=Success")
    print("\nStatus: ", end="", flush=True)
    
    try:
        while True:
            try:
                chunk = capture.audio_queue.get(timeout=1.0)
                
                # Check for silence vs speech
                if not chunk.is_speech:
                    print(".", end="", flush=True)
                    continue

                # If we get here, VAD detected speech
                print("?", end="", flush=True)
                
                if chunk.is_final:
                    result = stt.finalize()
                    if result and result.text:
                        print(f"\n\n[FINAL]: {result.text}\n", flush=True)
                        print("Status: ", end="", flush=True)
                    else:
                        print("!", end="", flush=True) # Final but no text found
                        
                else:
                    result = stt.process_chunk(chunk)
                    if result and result.text:
                        # Backspace over the '?' marks to show text
                        print(f"\r[Partial]: {result.text}...", end="", flush=True)
                        
            except Empty:
                pass
                
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        capture.stop()
        capture_thread.join()
        logger.info("Test complete.")

if __name__ == "__main__":
    main()