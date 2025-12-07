# Save this file as: src/pipeline/orchestrator.py

import threading
import queue
import time
import pyaudio
import numpy as np
import sounddevice as sd
from typing import Callable, Optional, Dict, Any

from src.audio.capture import StreamingAudioCapture
from src.stt.whisper_stt import WhisperSTT
from src.mt.helsinki_mt import HelsinkiMT
from src.tts.mms_tts import MMSTTS
from src.pipeline.message import PipelineMessage
from src.utils.logger import logger
from config.config import audio_config

class LiveTranslator:
    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "es",
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        device_index: Optional[int] = None,
        mic_sample_rate: Optional[int] = None,
    ):
        logger.info(f"Initializing Pipeline ({source_lang} -> {target_lang})...")
        self.event_sink = event_sink
        
        # 1. Audio Capture
        self.capture = StreamingAudioCapture(
            enable_vad=True,
            device_index=device_index,
            mic_sample_rate=mic_sample_rate,
            chunk_duration_ms=audio_config.chunk_duration_ms 
        )
        
        # 2. Speech-to-Text (Brain)
        if source_lang == "en":
            # CHANGE 'small.en' TO 'medium.en'
            stt_model = "medium.en" 
        else:
            # CHANGE 'small' TO 'medium'
            stt_model = "medium"
            
        self.stt = WhisperSTT(model_size=stt_model) 
        
        # 3. Translator
        self.mt = HelsinkiMT(source_lang=source_lang, target_lang=target_lang)
        
        # 4. Text-to-Speech
        lang_map = {"en": "eng", "es": "spa", "fr": "fra", "de": "deu"}
        tts_lang = lang_map.get(target_lang, "eng")
        self.tts = MMSTTS(lang=tts_lang)
        
        self.text_queue = queue.Queue()
        self.translation_queue = queue.Queue()
        self.audio_output_queue = queue.Queue()
        
        self.stop_event = threading.Event()
        self.threads = []
        
        self._warmup_models()
        self._emit("status", msg="warmup_complete", source_lang=source_lang, target_lang=target_lang)

    def _emit(self, event_type: str, **payload):
        if self.event_sink:
            try:
                self.event_sink({"type": event_type, **payload})
            except Exception as e:
                logger.debug(f"Event sink error: {e}")

    def _warmup_models(self):
        logger.info("Warming up AI models...")
        try:
            dummy_audio = np.zeros(16000, dtype=np.float32)
            self.stt.model.transcribe(dummy_audio, beam_size=1)
            self.mt.translate_text("warmup")
            self.tts.synthesize("warmup")
            logger.info("Warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup warning: {e}")

    def start(self):
        self.stop_event.clear()
        self.capture.start()
        
        self.threads = [
            threading.Thread(target=self.process_stt, name="STT_Worker"),
            threading.Thread(target=self.process_mt, name="MT_Worker"),
            threading.Thread(target=self.process_tts, name="TTS_Worker"),
            threading.Thread(target=self.process_output, name="Output_Worker")
        ]
        
        for t in self.threads:
            t.start()
            
        logger.info("System is LIVE! Speak into your microphone.")
        self._emit("status", msg="started")

    def stop(self):
        self.stop_event.set()
        self.capture.stop()
        for t in self.threads:
            t.join()
        logger.info("Pipeline Stopped.")
        self._emit("status", msg="stopped")

    def process_stt(self):
        while not self.stop_event.is_set():
            try:
                chunk = self.capture.audio_queue.get(timeout=0.5)
                if chunk.is_final:
                    result = self.stt.finalize()
                    if result and result.text:
                        self.text_queue.put(PipelineMessage(result.text, is_final=True))
                        print(f"\n[You]: {result.text}")
                        self._emit("stt_final", text=result.text)
                elif chunk.is_speech:
                    self.stt.process_chunk(chunk)
                    print(".", end="", flush=True)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"STT Error: {e}")
                self._emit("error", stage="stt", detail=str(e))

    def process_mt(self):
        while not self.stop_event.is_set():
            try:
                msg = self.text_queue.get(timeout=0.5)
                if msg.is_final:
                    result = self.mt.translate_text(msg.data, is_final=True)
                    if result and result.text:
                        self.translation_queue.put(PipelineMessage(result.text, is_final=True))
                        print(f"[Trans]: {result.text}")
                        self._emit("mt_final", text=result.text)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"MT Error: {e}")
                self._emit("error", stage="mt", detail=str(e))

    def process_tts(self):
        while not self.stop_event.is_set():
            try:
                msg = self.translation_queue.get(timeout=0.5)
                if msg.data:
                    audio_data = self.tts.synthesize(msg.data)
                    if len(audio_data) > 0:
                        self.audio_output_queue.put(PipelineMessage(audio_data, is_final=True))
                        self._emit("tts_ready", text=msg.data, duration=len(audio_data)/16000.0)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS Error: {e}")
                self._emit("error", stage="tts", detail=str(e))

    def process_output(self):
        while not self.stop_event.is_set():
            try:
                msg = self.audio_output_queue.get(timeout=0.5)
                if msg.data is not None and len(msg.data) > 0:
                    try:
                        audio = msg.data.astype(np.float32)

                        # Calculate duration and pad tiny clips to avoid driver hiccups
                        duration = len(audio) / 16000.0
                        if duration < 0.25:
                            padding = np.zeros(int(16000 * 0.25), dtype=np.float32)
                            audio = np.concatenate((audio, padding))
                            duration = len(audio) / 16000.0

                        # Non-blocking playback with a capped wait to avoid hangs
                        sd.play(audio, samplerate=16000, blocking=False)
                        self._emit("playback_start", duration=duration)

                        # Wait in short intervals so we can bail out if stop_event is set
                        max_wait = min(duration + 0.5, 6.0)  # never wait more than 6s
                        waited = 0.0
                        step = 0.05
                        while waited < max_wait and not self.stop_event.is_set():
                            time.sleep(step)
                            waited += step

                        sd.stop()
                        self._emit("playback_end")

                    except Exception as e:
                        logger.error(f"Playback Error: {e}")
                        sd.stop()
                        time.sleep(0.1)
                        self._emit("error", stage="playback", detail=str(e))
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Output Error: {e}")
                self._emit("error", stage="output", detail=str(e))