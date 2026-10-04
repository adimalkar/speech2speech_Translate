import numpy as np
import math
from typing import Tuple, Optional
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

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

def remove_dc_offset(audio_chunk: np.ndarray) -> np.ndarray:
    """
    Removes DC bias/offset from audio signal by subtracting the mean amplitude.
    Microphone hardware frequently introduces DC drift which biases RMS calculation
    and introduces audible artifacts during chunk stitching.
    """
    if len(audio_chunk) == 0:
        return audio_chunk
        
    chunk_float = audio_chunk.astype(np.float64)
    dc_free = chunk_float - np.mean(chunk_float)
    return np.clip(dc_free, -32768, 32767).astype(np.int16)

def calculate_zero_crossing_rate(audio_chunk: np.ndarray) -> float:
    """
    Calculate the Zero-Crossing Rate (ZCR) of an audio frame.
    Higher ZCR typically indicates unvoiced speech / fricatives or high-frequency noise,
    useful alongside RMS energy for robust speech discrimination.
    """
    if len(audio_chunk) < 2:
        return 0.0
        
    signs = np.sign(audio_chunk)
    # Replace 0s with 1 to avoid false crossings
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings) / float(len(audio_chunk) - 1)


def detect_audio_clipping(
    audio_chunk: np.ndarray,
    clip_threshold: int = 32700,
    max_clipped_ratio: float = 0.01
) -> Tuple[bool, float]:
    """
    Detect digital clipping / saturation in an audio buffer.
    
    Args:
        audio_chunk (np.ndarray): 1D array of 16-bit PCM audio samples.
        clip_threshold (int): Absolute amplitude threshold indicating clipping (max 32767).
        max_clipped_ratio (float): Maximum tolerable fraction of clipped samples (default 1%).
        
    Returns:
        Tuple[bool, float]: (is_clipped, clipped_ratio)
    """
    if len(audio_chunk) == 0:
        return False, 0.0
    
    clipped_count = np.sum(np.abs(audio_chunk) >= clip_threshold)
    clipped_ratio = float(clipped_count) / float(len(audio_chunk))
    is_clipped = clipped_ratio > max_clipped_ratio
    
    if is_clipped:
        logger.warning(
            f"Audio clipping detected: {clipped_ratio * 100.0:.2f}% samples exceed threshold {clip_threshold}"
        )
    return is_clipped, round(clipped_ratio, 4)


def apply_pre_emphasis(audio_chunk: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Apply pre-emphasis high-pass filter: y[n] = x[n] - coeff * x[n-1].
    Balances high-frequency formants and attenuates low-frequency microphone rumble,
    standard in speech recognition front-end preprocessing.
    """
    if len(audio_chunk) <= 1:
        return audio_chunk
    
    chunk_float = audio_chunk.astype(np.float64)
    emphasized = np.empty_like(chunk_float)
    emphasized[0] = chunk_float[0]
    emphasized[1:] = chunk_float[1:] - coeff * chunk_float[:-1]
    
    return np.clip(emphasized, -32768, 32767).astype(np.int16)


def resample_linear(audio_chunk: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Lightweight, dependency-free 1D linear interpolation resampler.
    Converts audio between sampling rates (e.g., 44.1/48 kHz to 16 kHz) for STT models.
    """
    if orig_sr == target_sr or len(audio_chunk) == 0:
        return audio_chunk
    
    orig_length = len(audio_chunk)
    target_length = int(round(orig_length * float(target_sr) / float(orig_sr)))
    if target_length <= 0:
        return np.array([], dtype=audio_chunk.dtype)
    
    orig_indices = np.linspace(0, orig_length - 1, num=orig_length)
    target_indices = np.linspace(0, orig_length - 1, num=target_length)
    
    resampled_float = np.interp(target_indices, orig_indices, audio_chunk.astype(np.float64))
    return np.clip(resampled_float, -32768, 32767).astype(np.int16)

