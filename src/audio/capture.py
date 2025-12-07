# Save this file as: src/audio/capture.py

import pyaudio
import numpy as np
from threading import Thread, Event
import queue  # Ensure this is imported!
import time
from typing import Optional
import librosa

from src.audio.chunk import AudioChunk, AudioBuffer
from src.audio.vad import StreamingVAD
from src.utils.logger import logger

class StreamingAudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 500,
        audio_queue: Optional[queue.Queue] = None,
        enable_vad: bool = True,
        device_index: Optional[int] = None,
        mic_sample_rate: Optional[int] = None
    ):
        self.target_rate = sample_rate
        self.mic_rate = int(mic_sample_rate) if mic_sample_rate else None
        self.device_index = device_index
        self.chunk_size = int(self.target_rate * chunk_duration_ms / 1000)
        
        self.audio_queue = audio_queue or queue.Queue(maxsize=10) 
        self.enable_vad = enable_vad
        
        self.stop_event = Event()
        self.audio_buffer = AudioBuffer(self.chunk_size)
        self.chunk_counter = 0
        
        # VAD Settings tuned for lower latency while keeping stability:
        # - speech_threshold=0.2 : sensitive to short/quiet speech
        # - silence_duration_ms=900 : ~0.9s pause before finalize (faster commit)
        self.vad = StreamingVAD(
            sample_rate=self.target_rate,
            frame_duration_ms=32, 
            silence_duration_ms=900,
            speech_threshold=0.2
        ) if enable_vad else None
        
        self.p = None
        self.stream = None
    
    def start(self) -> Thread:
        self.stop_event.clear()
        thread = Thread(target=self._capture_loop, daemon=True, name='AudioCapture')
        thread.start()
        logger.info(f"Capture started. Mic: {self.mic_rate}Hz -> AI: {self.target_rate}Hz")
        return thread
    
    def _capture_loop(self):
        self.p = pyaudio.PyAudio()
        try:
            # Auto-select default input device if not provided
            if self.device_index is None:
                try:
                    default_info = self.p.get_default_input_device_info()
                    self.device_index = default_info.get("index", None)
                    if self.mic_rate is None:
                        self.mic_rate = int(default_info.get("defaultSampleRate", 48000))
                    logger.info(f"Auto-selected input device {self.device_index} @ {self.mic_rate}Hz")
                except Exception as e:
                    logger.warning(f"Could not get default input device, falling back. {e}")
                    self.device_index = None
                    if self.mic_rate is None:
                        self.mic_rate = 48000

            # If mic_rate still None, set a safe default
            if self.mic_rate is None:
                self.mic_rate = 48000

            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.mic_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=1024
            )
            
            logger.info(f"PyAudio stream opened on device {self.device_index}")
            
            while not self.stop_event.is_set():
                try:
                    data = self.stream.read(1024, exception_on_overflow=False)
                    raw_audio = np.frombuffer(data, dtype=np.float32)
                    
                    # Manual Volume Boost (Software Gain)
                    # If your mic is quiet, we multiply the signal to help VAD detect it.
                    raw_audio = raw_audio * 2.0 

                    if self.mic_rate != self.target_rate:
                        processed_audio = librosa.resample(raw_audio, orig_sr=self.mic_rate, target_sr=self.target_rate)
                    else:
                        processed_audio = raw_audio

                    self.audio_buffer.add(processed_audio)
                    
                    if self.audio_buffer.full:
                        self._process_and_emit_chunk()
                        self.audio_buffer.reset()
                        
                except Exception as e:
                    logger.error(f"Error reading/resampling: {e}")
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Critical capture error: {e}")
        finally:
            self._cleanup()
    
    def _process_and_emit_chunk(self):
        chunk_audio = self.audio_buffer.get_chunk()
        
        is_speech = False
        confidence = 0.0
        is_final = False
        
        if self.vad:
            step_size = 512 
            for i in range(0, len(chunk_audio), step_size):
                frame = chunk_audio[i : i + step_size]
                if len(frame) < step_size:
                    frame = np.pad(frame, (0, step_size - len(frame)))
                
                is_speech_frame, conf = self.vad.process_frame(frame)
                if is_speech_frame:
                    is_speech = True
                    confidence = max(confidence, conf)
            
            is_final = self.vad.should_end_utterance()
        else:
            is_speech = True
            confidence = 1.0
        
        # Don't queue silence unless it's an "End of Sentence" marker
        if not is_speech and not is_final:
            return

        self.chunk_counter += 1
        chunk = AudioChunk(
            chunk_id=self.chunk_counter,
            audio_data=chunk_audio,
            timestamp=time.time(),
            duration_ms=self.chunk_size / self.target_rate * 1000,
            sample_rate=self.target_rate,
            vad_confidence=confidence,
            is_speech=is_speech,
            is_final=is_final
        )
        
        if is_final and self.vad:
            self.vad.reset()
        
        if self.audio_queue.full():
            try:
                self.audio_queue.get_nowait() # Discard oldest
            except queue.Empty:
                pass
        
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            pass 

    def stop(self):
        self.stop_event.set()
    
    def _cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        logger.info("Microphone capture stopped")