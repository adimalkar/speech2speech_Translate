# Save this file as: tests/test_vad_integration.py

import librosa
import numpy as np
import os
from datasets import load_dataset
from src.utils.logger import logger
from src.audio.vad import StreamingVAD
from config.config import audio_config

# --- 1. Load the VAD ---
vad = StreamingVAD(
    sample_rate=audio_config.sample_rate,
    frame_duration_ms=audio_config.vad_frame_duration_ms,
    silence_duration_ms=400
)
logger.info("StreamingVAD initialized.")

# --- 2. Load Data (Robust Method) ---
logger.info("Loading 'es_en' test file info...")
audio_data = None

try:
    # Load dataset in streaming mode just to get the path quickly
    dataset = load_dataset(
        "fixie-ai/covost2", 
        "es_en", 
        split="test",
        streaming=True 
    )
    
    # Get the first example
    example = next(iter(dataset))
    
    # EXTRACT THE PATH SAFELY
    # The dataset returns a path relative to the cache directory.
    # We need to handle this carefully.
    raw_path = example['file']
    
    # If it's an absolute path that doesn't exist (the /home/farzad bug),
    # we need to find where it actually is in YOUR cache.
    if not os.path.exists(raw_path):
        # Common trick: The file is actually inside the 'audio' dictionary if we don't stream
        # But since streaming is safer for big datasets, we'll try a hybrid approach.
        logger.info("Original path not found. Searching in cache...")
        
        # Reload without streaming to get the local path resolved by huggingface
        ds_local = load_dataset("fixie-ai/covost2", "es_en", split="test", streaming=False)
        local_example = ds_local[0]
        
        # In the local version, 'audio' is a dictionary with 'path'
        # AND 'array'. We can just use the array directly!
        audio_data = local_example['audio']['array']
        sampling_rate = local_example['audio']['sampling_rate']
        
        logger.info(f"Loaded audio directly from dataset array. Native SR: {sampling_rate}")
        
        # Resample if necessary
        if sampling_rate != audio_config.sample_rate:
            logger.info(f"Resampling from {sampling_rate} to {audio_config.sample_rate}...")
            audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=audio_config.sample_rate)

except Exception as e:
    logger.error(f"Critical error loading data: {e}")
    exit(1) # Stop here if data fails!

# --- 3. Simulate Streaming ---
if audio_data is not None:
    logger.info(f"Audio ready. Shape: {audio_data.shape}")
    logger.info("Simulating audio stream... (S = Speech, _ = Silence, ! = Utterance End)")

    # Ensure float32
    audio_data = audio_data.astype(np.float32)

    # Get the VAD's frame size
    frame_size = vad.frame_size
    total_frames = 0
    speech_frames = 0
    output_display = ""

    for i in range(0, len(audio_data), frame_size):
        # Get frame
        frame = audio_data[i : i + frame_size]
        
        # Pad last frame if needed
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
            
        # Process
        try:
            is_speech, confidence = vad.process_frame(frame)
            total_frames += 1
            
            if is_speech:
                speech_frames += 1
                output_display += "S"
            else:
                output_display += "_"

            if vad.should_end_utterance():
                output_display += "!\n"
                vad.reset()
            
            # Print chunk
            if len(output_display) > 80:
                print(output_display)
                output_display = ""
                
        except Exception as e:
            logger.error(f"VAD Error on frame {i}: {e}")
            break

    if output_display:
        print(output_display)

    logger.info("\n--- Simulation Complete ---")
    logger.info(f"Processed {total_frames} frames.")
    logger.info(f"Detected {speech_frames} speech frames.")

else:
    logger.error("No audio data loaded. Exiting.")