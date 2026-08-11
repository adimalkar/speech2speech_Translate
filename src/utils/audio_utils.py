import numpy as np
import math
from typing import Tuple, Optional
from loguru import logger

def calculate_rms_energy(audio_chunk: np.ndarray) -> float:
    """
    Calculate the Root Mean Square (RMS) energy of an audio chunk.
    This is critical for Voice Activity Detection (VAD) to filter out background silence
    and prevent the STT engine from transcribing empty noise.

    Args:
        audio_chunk (np.ndarray): 1D array of audio samples (usually int16 or float32)

    Returns:
        float: The RMS energy level of the chunk
    """
    if len(audio_chunk) == 0:
        return 0.0
    
    # Cast to float64 to prevent overflow during squaring
    chunk_float = audio_chunk.astype(np.float64)
    mean_square = np.mean(chunk_float ** 2)
    
    return math.sqrt(mean_square)

def normalize_audio(audio_chunk: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """
    Normalize audio chunk to a target dBFS (Decibels relative to full scale).
    Ensures that microphone input is consistently leveled before hitting the 
    STT (Speech-to-Text) inference model, improving transcription accuracy.

    Args:
        audio_chunk (np.ndarray): The raw audio input array
        target_dbfs (float): Target volume in decibels (default -20.0 dBFS)

    Returns:
        np.ndarray: The gain-normalized audio array
    """
    rms = calculate_rms_energy(audio_chunk)
    if rms == 0.0:
        return audio_chunk
        
    # Convert RMS to dBFS
    # Assuming standard 16-bit PCM where max amplitude is 32768
    current_dbfs = 20 * math.log10(rms / 32768.0)
    
    # Calculate required gain multiplier
    gain_db = target_dbfs - current_dbfs
    gain_multiplier = 10 ** (gain_db / 20.0)
    
    # Apply gain and clip to prevent integer overflow clipping distortion
    normalized = audio_chunk * gain_multiplier
    normalized = np.clip(normalized, -32768, 32767)
    
    return normalized.astype(np.int16)

def is_speech_active(audio_chunk: np.ndarray, threshold_rms: float = 500.0) -> bool:
    """
    Fast, lightweight gatekeeper to determine if a chunk contains active speech.
    
    Args:
        audio_chunk (np.ndarray): 1D array of audio samples
        threshold_rms (float): The minimum RMS energy required to trigger speech
        
    Returns:
        bool: True if speech is detected, False otherwise
    """
    energy = calculate_rms_energy(audio_chunk)
    is_active = energy > threshold_rms
    
    if not is_active:
        logger.debug(f"Audio chunk rejected by VAD gate (Energy: {energy:.2f} < {threshold_rms})")
        
    return is_active
