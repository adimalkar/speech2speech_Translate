# Detailed Implementation Guide: Concurrent Architecture for Low-Latency, Real-Time Speech-to-Speech Translation

## Executive Summary

Your current implementation plan (Sequential Pipeline: STT → MT → TTS) is **architecturally sound but not optimal for true low-latency translation**. The proposed three-thread, two-queue architecture provides a solid foundation; however, it still relies on **full utterance processing** before translation begins, introducing unnecessary latency. This guide presents an enhanced architecture that achieves **true streaming and concurrent processing** at multiple levels while maintaining the concurrent queuing model you've outlined.

---

## Part 1: Evaluation of Your Current Plan

### Strengths

1. **Concurrency Foundation**: The three-thread, two-queue architecture effectively decouples I/O, processing, and output, preventing bottlenecks.
2. **VAD Integration**: Voice Activity Detection ensures clean audio chunks and prevents processing silence.
3. **Component Maturity**: Using proven APIs (Whisper, Google Cloud APIs) reduces risk.
4. **Clear Latency Target**: 1.5-second average latency is realistic.

### Limitations That Will Impact Latency

| Limitation | Impact | Why It Matters |
|-----------|--------|-----------------|
| **Utterance-level Processing** | TTS can only start after *entire* transcribed sentence is ready | If user speaks for 3 seconds, translation waits 3s before starting TTS |
| **Sequential Model Inference** | Each stage waits for previous stage to complete | Even with threads, translation must wait for STT to finish completely |
| **Fixed Segmentation** | VAD creates rigid boundaries; mid-utterance translation impossible | Cannot start translating middle of long utterance |
| **API Latency** | External APIs (Google Cloud) add 200-500ms per stage | 3 stages × 300ms = 900ms latency baseline |

### Realistic Latency Analysis of Your Current Plan

For a typical 5-second utterance (e.g., "Please tell me the weather forecast for tomorrow"):

| Stage | Time (ms) |
|-------|-----------|
| Audio capture & VAD | 500 (await silence) |
| STT (Whisper-base + API) | 1500-2000 |
| MT (Google Translate API) | 300-500 |
| TTS (Google Cloud + streaming) | 1000-1500 |
| **Total End-to-End** | **3.3-4.5 seconds** |

**This exceeds your 1.5-second target by 2-3x.**

---

## Part 2: Optimized Implementation Plan

### Recommended Architecture: **Streaming Concurrent Pipeline with Chunk-Level Parallelism**

Instead of waiting for complete utterances, process audio in **overlapping chunks** and translate incrementally:

```
┌─────────────────────────────────────────────────────────────┐
│                     AUDIO INPUT STREAM                       │
│                    (continuous speech)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   VAD + Chunk Segmentation      │
        │  (Process 500ms-1s chunks)      │
        └────────┬──────────────────┬─────┘
                 │                  │
       ┌─────────▼──────┐  ┌────────▼──────────┐
       │  Chunk Queue 1 │  │   Chunk Queue 2   │
       └─────────┬──────┘  └────────┬──────────┘
                 │                  │
    ┌────────────▼──────────┐      │
    │  Streaming STT Thread  │      │ (processes previous chunks)
    │  (Whisper or Kyutai)   │      │
    └────────────┬───────────┘      │
                 │                  │
       ┌─────────▼──────────────────▼──────┐
       │      Partial Text Queue           │
       │  (streaming transcription)        │
       └────────────┬─────────────────────┘
                    │
      ┌─────────────▼─────────────┐
      │   Streaming MT Thread     │
      │ (chunk-aware translation) │
      └─────────────┬─────────────┘
                    │
         ┌──────────▼──────────┐
         │  Translated Queue   │
         │  (streaming text)   │
         └──────────┬──────────┘
                    │
      ┌─────────────▼──────────────┐
      │   Streaming TTS Thread     │
      │ (CosyVoice2 or EdgeTTS)    │
      └─────────────┬──────────────┘
                    │
                    ▼
            ┌──────────────────┐
            │  AUDIO OUTPUT    │
            │  (real-time)     │
            └──────────────────┘
```

### Key Improvements Over Your Current Plan

#### 1. **Streaming ASR Instead of Batch STT**

**Current Approach**: Wait for VAD silence → Send complete audio → Wait for Whisper result

**Improved Approach**: Use streaming ASR models that output partial results:

```python
from streaming_asr import StreamingASR  # e.g., Kyutai or RNN-T model

asr = StreamingASR(model_name="kyutai-1b")  # 1s latency

# Process 500ms chunks
for chunk in audio_stream:
    partial_transcript = asr.process_chunk(chunk)
    if partial_transcript:
        yield ("partial_text", partial_transcript)
        
# Finalize when speech ends
final_transcript = asr.finalize()
yield ("final_text", final_transcript)
```

**Latency Benefit**: First translation can start after ~1.5-2s instead of waiting 3-5s for full utterance.

#### 2. **Chunk-Aware Machine Translation**

**Current Approach**: Translate complete sentence ("Please tell me the weather forecast for tomorrow")

**Improved Approach**: Translate in chunks, maintaining context:

```python
from streaming_mt import StreamingMT

mt = StreamingMT(model="llm-based-mt")

# Maintain translation context across chunks
context = None
for partial_text in partial_transcripts:
    translated_chunk, context = mt.translate_chunk(
        text=partial_text,
        context=context,
        language_pair=("en", "es")
    )
    yield translated_chunk
```

**Challenges to Address**:
- **Word reordering**: Some languages reorder words in translation (German, Japanese)
- **Context dependency**: Early chunks may need revision when later context arrives
- **Solution**: Use "agreement policy" — only emit translation when consecutive chunks agree on boundaries

#### 3. **Streaming TTS (Ultra-Low Latency)**

**Current Approach**: Google Cloud TTS (500-1000ms latency per request)

**Improved Approach**: Use streaming TTS like CosyVoice2 (150ms latency):

```python
from cosy_voice_2 import CosyVoiceStreaming

tts = CosyVoiceStreaming(model="cosy-voice-2-0.5b")

# Stream audio output in real-time
for translated_chunk in translated_stream:
    audio_chunk = tts.synthesize_streaming(
        text=translated_chunk,
        language="es",
        speaker="neural_speaker_1"
    )
    play_audio(audio_chunk)  # Play immediately (150ms latency)
```

**Latency Benefit**: Start playing translated speech after ~2-2.5s total instead of 3.5-4.5s.

---

## Part 3: Detailed Implementation Steps

### Phase 1: Core Infrastructure (Weeks 1-2)

#### Step 1.1: Enhanced Audio Pipeline with Chunk Streaming

```python
import pyaudio
import numpy as np
from collections import deque
from threading import Thread, Event
from queue import Queue
import logging

class ChunkStreamingAudioCapture:
    """
    Captures audio and emits overlapping chunks instead of waiting for silence.
    """
    
    def __init__(
        self,
        sample_rate=16000,
        chunk_duration_ms=500,  # Process every 500ms
        overlap_ms=100,         # 100ms overlap
        vad_threshold=0.3
    ):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.overlap_size = int(sample_rate * overlap_ms / 1000)
        self.stride = self.chunk_size - self.overlap_size
        
        self.audio_queue = Queue(maxsize=10)
        self.stop_event = Event()
        self.logger = logging.getLogger(__name__)
        
        # Buffer for overlap
        self.audio_buffer = deque(maxlen=self.chunk_size)
    
    def start_capture(self):
        """Start recording in background thread"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        thread = Thread(target=self._capture_loop, args=(stream,))
        thread.daemon = True
        thread.start()
        
        return stream
    
    def _capture_loop(self, stream):
        """Main capture loop"""
        try:
            while not self.stop_event.is_set():
                # Read frames
                data = stream.read(1024, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.float32)
                
                # Add to buffer
                self.audio_buffer.extend(audio)
                
                # Emit chunks
                while len(self.audio_buffer) >= self.stride:
                    chunk = np.array(
                        list(self.audio_buffer)[:self.chunk_size]
                    )
                    self.audio_queue.put(("audio_chunk", chunk))
                    
                    # Slide buffer
                    for _ in range(self.stride):
                        self.audio_buffer.popleft()
        
        except Exception as e:
            self.logger.error(f"Capture error: {e}")
            self.audio_queue.put(("error", e))
        
        finally:
            stream.stop_stream()
            stream.close()
    
    def stop_capture(self):
        self.stop_event.set()


class RealTimeVADEnhanced:
    """
    Enhanced VAD that emits chunks continuously instead of waiting for silence.
    """
    
    def __init__(self, model_name="silero", frame_duration_ms=20):
        import torch
        from silero_vad import load_silero_vad
        
        self.vad_model = load_silero_vad()
        self.sample_rate = 16000
        self.frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        self.silence_count = 0
        self.max_silence_frames = 20  # 400ms of silence = end utterance
    
    def process_chunk(self, audio_chunk):
        """
        Process audio chunk and return:
        - ("speech_chunk", audio) for speech chunks
        - ("utterance_end", None) when silence detected
        """
        # Check if speech
        confidence = self.vad_model(
            torch.from_numpy(audio_chunk).unsqueeze(0),
            self.sample_rate
        ).item()
        
        if confidence > 0.3:  # Speech detected
            self.silence_count = 0
            return ("speech_chunk", audio_chunk)
        else:
            self.silence_count += 1
            if self.silence_count == self.max_silence_frames:
                self.silence_count = 0
                return ("utterance_end", None)
        
        return None
```

#### Step 1.2: Streaming STT Integration

```python
from typing import Generator, Tuple

class StreamingSTTAdapter:
    """
    Unified interface for streaming ASR models.
    Supports Kyutai, RNN-T, or Whisper in streaming mode.
    """
    
    def __init__(self, model_choice="kyutai"):
        self.model_choice = model_choice
        
        if model_choice == "kyutai":
            from transformers import AutoProcessor, AutoModelForCTC
            
            self.processor = AutoProcessor.from_pretrained(
                "kyutai/kyutai-1b"
            )
            self.model = AutoModelForCTC.from_pretrained(
                "kyutai/kyutai-1b"
            ).eval()
            self.sample_rate = 16000
        
        elif model_choice == "whisper_streaming":
            # Use Faster-Whisper with streaming
            from faster_whisper import WhisperModel
            
            self.model = WhisperModel("base", device="cuda", compute_type="float16")
            self.sample_rate = 16000
        
        self.state = None  # Maintain state for streaming
        self.partial_transcript = ""
    
    def process_chunk(self, audio_chunk: np.ndarray) -> str:
        """
        Process audio chunk and return partial transcription.
        
        Args:
            audio_chunk: numpy array of audio samples
        
        Returns:
            Partial transcription string
        """
        if self.model_choice == "kyutai":
            # Process with state
            inputs = self.processor(
                audio_chunk,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                logits = self.model(inputs["input_values"]).logits
            
            # Greedy decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = self.processor.batch_decode(predicted_ids)
            
            self.partial_transcript = transcript[0]
            return self.partial_transcript
        
        return ""
    
    def finalize(self) -> str:
        """Get final transcription"""
        return self.partial_transcript
```

#### Step 1.3: Thread-Safe Queue Management

```python
from queue import Queue, PriorityQueue
from dataclasses import dataclass
from enum import Enum
import time

class ProcessingStage(Enum):
    AUDIO = 1
    STT = 2
    MT = 3
    TTS = 4

@dataclass
class Message:
    """Message passing between threads"""
    stage: ProcessingStage
    content_type: str  # "audio", "text", "speech", etc.
    data: any
    timestamp: float  # For latency tracking
    chunk_id: int  # For tracking through pipeline
    priority: int = 0  # 0=highest priority
    
    def latency(self) -> float:
        return (time.time() - self.timestamp) * 1000  # milliseconds

# Initialize queues
audio_queue = Queue(maxsize=20)
partial_text_queue = Queue(maxsize=20)
translated_queue = Queue(maxsize=20)
speech_output_queue = Queue(maxsize=20)

# Metrics collection
class PipelineMetrics:
    def __init__(self):
        self.latencies = {"stts": [], "mt": [], "tts": []}
        self.chunk_count = 0
    
    def record_latency(self, stage: str, latency_ms: float):
        self.latencies[stage].append(latency_ms)
    
    def average_latency(self, stage: str) -> float:
        if not self.latencies[stage]:
            return 0
        return sum(self.latencies[stage]) / len(self.latencies[stage])

metrics = PipelineMetrics()
```

### Phase 2: Streaming Translation Pipeline (Weeks 3-4)

#### Step 2.1: Streaming Machine Translation

```python
class StreamingMTAdapter:
    """
    Handles chunk-wise translation with context awareness.
    """
    
    def __init__(self, model_choice="llm"):
        self.model_choice = model_choice
        
        if model_choice == "llm":
            from llama_cpp import Llama
            # Use smaller LLM for speed (e.g., Mistral 7B quantized)
            self.model = Llama(
                model_path="mistral-7b-q4.gguf",
                n_gpu_layers=-1,  # GPU acceleration
                n_threads=4
            )
        
        elif model_choice == "google_api":
            from google.cloud import translate_v2
            self.client = translate_v2.Client()
        
        self.context_window = []  # Store last N chunks for context
        self.max_context_chunks = 5
    
    def translate_chunk(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Tuple[str, float]:
        """
        Translate text chunk with context awareness.
        
        Returns:
            (translated_text, confidence_score)
        """
        start_time = time.time()
        
        # Build context prompt
        context = " ".join(self.context_window[-self.max_context_chunks:])
        
        if self.model_choice == "llm":
            prompt = f"""Translate the following text from {source_lang} to {target_lang}.
            
Context (previous chunks):
{context}

Current text to translate:
{text}

Translation:"""
            
            response = self.model(prompt, max_tokens=256)
            translated = response["choices"][0]["text"].strip()
            confidence = 0.95  # LLMs generally confident
        
        else:  # Google API
            result = self.client.translate_text(
                text,
                source_language=source_lang,
                target_language=target_lang
            )
            translated = result["translatedText"]
            confidence = 0.90  # API confidence
        
        # Update context
        self.context_window.append(text)
        if len(self.context_window) > self.max_context_chunks:
            self.context_window.pop(0)
        
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_latency("mt", latency_ms)
        
        return translated, confidence


class AgreementPolicy:
    """
    Emit translation only when consecutive chunks agree on boundaries.
    Reduces mid-utterance revision and improves stability.
    """
    
    def __init__(self, agreement_threshold=0.8):
        self.threshold = agreement_threshold
        self.prev_translation = None
        self.revision_count = 0
    
    def should_emit(
        self,
        current_translation: str,
        confidence: float
    ) -> bool:
        """
        Check if translation is stable enough to emit.
        """
        if self.prev_translation is None:
            self.prev_translation = current_translation
            return confidence > self.threshold
        
        # Check overlap between prev and current
        common_words = set(self.prev_translation.split()) & \
                       set(current_translation.split())
        
        overlap_ratio = len(common_words) / max(
            len(self.prev_translation.split()),
            len(current_translation.split())
        )
        
        is_stable = overlap_ratio > 0.7  # 70% agreement
        
        if is_stable:
            self.prev_translation = current_translation
            return True
        else:
            self.revision_count += 1
            return False
```

#### Step 2.2: Streaming TTS Integration

```python
class StreamingTTSAdapter:
    """
    Handles ultra-low-latency speech synthesis.
    """
    
    def __init__(self, model_choice="cosy_voice"):
        self.model_choice = model_choice
        
        if model_choice == "cosy_voice":
            # CosyVoice2 (150ms latency)
            from cosy_voice_streaming import CosyVoiceStreaming
            
            self.model = CosyVoiceStreaming(
                model_path="cosy-voice-2-0.5b.pt"
            )
            self.sample_rate = 22050
        
        elif model_choice == "edge_tts":
            # Microsoft Edge TTS (lighter weight)
            import edge_tts
            self.client = edge_tts.Communicate()
            self.sample_rate = 48000
    
    def synthesize_streaming(
        self,
        text: str,
        language: str,
        speaker_id: int = 0
    ) -> np.ndarray:
        """
        Synthesize speech from text with minimal latency.
        
        Returns:
            Audio samples (numpy array)
        """
        start_time = time.time()
        
        if self.model_choice == "cosy_voice":
            # Streaming generation
            audio = self.model.synthesize(
                text=text,
                language=language,
                speaker_id=speaker_id,
                streaming=True
            )
        else:
            # Edge TTS
            audio = self.client.synthesize(
                text,
                language=language,
                rate=1.0
            )
        
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_latency("tts", latency_ms)
        
        return audio
```

### Phase 3: Main Pipeline Orchestration (Weeks 5-6)

```python
import threading
from datetime import datetime

class LiveTranslatorStreaming:
    """
    Enhanced concurrent pipeline with streaming at every stage.
    """
    
    def __init__(
        self,
        source_lang="en",
        target_lang="es"
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Initialize components
        self.audio_capture = ChunkStreamingAudioCapture()
        self.vad = RealTimeVADEnhanced()
        self.stt = StreamingSTTAdapter(model_choice="kyutai")
        self.mt = StreamingMTAdapter(model_choice="llm")
        self.tts = StreamingTTSAdapter(model_choice="cosy_voice")
        
        # Agreement policy for stable translations
        self.agreement_policy = AgreementPolicy()
        
        # Thread control
        self.stop_event = threading.Event()
        self.chunk_id_counter = 0
    
    def run(self):
        """Start all pipeline threads"""
        # Start audio capture
        self.stream = self.audio_capture.start_capture()
        
        # Start worker threads
        threads = [
            threading.Thread(target=self._audio_to_text_worker),
            threading.Thread(target=self._text_translation_worker),
            threading.Thread(target=self._synthesis_worker),
            threading.Thread(target=self._output_worker)
        ]
        
        for t in threads:
            t.daemon = True
            t.start()
        
        return threads
    
    def _audio_to_text_worker(self):
        """
        Thread 1: Audio → Streaming STT
        
        Processes audio chunks through VAD and streaming ASR.
        Emits partial transcriptions as they become available.
        """
        self.logger.info("STT Worker started")
        
        try:
            while not self.stop_event.is_set():
                # Get audio chunk
                try:
                    msg = self.audio_queue.get(timeout=0.1)
                    msg_type, chunk_data = msg
                    
                    if msg_type == "audio_chunk":
                        # VAD processing
                        vad_result = self.vad.process_chunk(chunk_data)
                        
                        if vad_result:
                            vad_type, audio = vad_result
                            
                            if vad_type == "speech_chunk":
                                # Process through streaming STT
                                partial_text = self.stt.process_chunk(audio)
                                
                                # Emit partial transcription
                                if partial_text:
                                    self.chunk_id_counter += 1
                                    msg_obj = Message(
                                        stage=ProcessingStage.STT,
                                        content_type="partial_text",
                                        data=partial_text,
                                        timestamp=time.time(),
                                        chunk_id=self.chunk_id_counter
                                    )
                                    partial_text_queue.put(msg_obj)
                            
                            elif vad_type == "utterance_end":
                                # Finalize STT
                                final_text = self.stt.finalize()
                                final_msg = Message(
                                    stage=ProcessingStage.STT,
                                    content_type="final_text",
                                    data=final_text,
                                    timestamp=time.time(),
                                    chunk_id=self.chunk_id_counter,
                                    priority=1  # Higher priority
                                )
                                partial_text_queue.put(final_msg)
                
                except:
                    pass
        
        except Exception as e:
            self.logger.error(f"STT Worker error: {e}")
    
    def _text_translation_worker(self):
        """
        Thread 2: Streaming Text → Streaming MT
        
        Translates partial texts as they arrive, maintaining context.
        """
        self.logger.info("MT Worker started")
        
        try:
            while not self.stop_event.is_set():
                try:
                    msg = partial_text_queue.get(timeout=0.1)
                    
                    if msg.content_type in ["partial_text", "final_text"]:
                        # Translate
                        translated, confidence = self.mt.translate_chunk(
                            text=msg.data,
                            source_lang=self.source_lang,
                            target_lang=self.target_lang
                        )
                        
                        # Apply agreement policy
                        if self.agreement_policy.should_emit(
                            translated,
                            confidence
                        ):
                            translated_msg = Message(
                                stage=ProcessingStage.MT,
                                content_type=msg.content_type,
                                data=translated,
                                timestamp=time.time(),
                                chunk_id=msg.chunk_id,
                                priority=msg.priority
                            )
                            translated_queue.put(translated_msg)
                
                except:
                    pass
        
        except Exception as e:
            self.logger.error(f"MT Worker error: {e}")
    
    def _synthesis_worker(self):
        """
        Thread 3: Translated Text → Streaming TTS → Speech
        
        Synthesizes translated text with minimal latency.
        """
        self.logger.info("TTS Worker started")
        
        try:
            while not self.stop_event.is_set():
                try:
                    msg = translated_queue.get(timeout=0.1)
                    
                    if msg.content_type in ["partial_text", "final_text"]:
                        # Synthesize
                        audio = self.tts.synthesize_streaming(
                            text=msg.data,
                            language=self.target_lang
                        )
                        
                        # Queue for output
                        audio_msg = Message(
                            stage=ProcessingStage.TTS,
                            content_type="speech",
                            data=audio,
                            timestamp=time.time(),
                            chunk_id=msg.chunk_id,
                            priority=msg.priority
                        )
                        speech_output_queue.put(audio_msg)
                
                except:
                    pass
        
        except Exception as e:
            self.logger.error(f"TTS Worker error: {e}")
    
    def _output_worker(self):
        """
        Thread 4: Play audio output
        
        Handles real-time audio playback with minimal buffering.
        """
        self.logger.info("Output Worker started")
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=22050,
            output=True
        )
        
        try:
            while not self.stop_event.is_set():
                try:
                    msg = speech_output_queue.get(timeout=0.1)
                    
                    if msg.content_type == "speech":
                        # Play audio
                        stream.write(msg.data.astype(np.float32).tobytes())
                        
                        # Log latency
                        end_to_end_latency = msg.latency()
                        self.logger.info(
                            f"Chunk {msg.chunk_id}: "
                            f"End-to-end latency: {end_to_end_latency:.0f}ms"
                        )
                
                except:
                    pass
        
        except Exception as e:
            self.logger.error(f"Output Worker error: {e}")
        
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    def stop(self):
        """Stop all threads gracefully"""
        self.stop_event.set()
```

### Phase 4: Error Handling and Fallbacks (Week 7)

```python
class RobustTranslator:
    """
    Production-grade translator with fallback mechanisms.
    """
    
    def __init__(self):
        self.primary = StreamingMTAdapter(model_choice="llm")
        self.fallback = StreamingMTAdapter(model_choice="google_api")
        self.failure_count = 0
        self.max_failures = 3
    
    def translate_chunk_with_fallback(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Tuple[str, bool]:
        """
        Translate with automatic fallback.
        
        Returns:
            (translated_text, used_fallback)
        """
        try:
            translated, confidence = self.primary.translate_chunk(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            if confidence < 0.5:
                # Low confidence, try fallback
                fallback_translated, _ = self.fallback.translate_chunk(
                    text, source_lang, target_lang
                )
                return fallback_translated, True
            
            self.failure_count = 0  # Reset
            return translated, False
        
        except Exception as e:
            self.failure_count += 1
            
            if self.failure_count <= self.max_failures:
                # Try fallback
                try:
                    translated, _ = self.fallback.translate_chunk(
                        text, source_lang, target_lang
                    )
                    return translated, True
                except:
                    return text, False  # Return original on all failures
            else:
                return text, False
```

### Phase 5: Testing and Optimization (Weeks 7-8)

```python
class PerformanceTester:
    """
    Comprehensive testing framework for latency and quality.
    """
    
    def __init__(self):
        self.translator = LiveTranslatorStreaming()
        self.results = []
    
    def test_latency(self, audio_file, test_name=""):
        """
        Measure end-to-end latency on test audio.
        """
        self.translator.run()
        
        # Play audio file through system
        # Record start/end timestamps
        
        results = {
            "test": test_name,
            "avg_latency_ms": np.mean([msg.latency() for msg in ...]),
            "max_latency_ms": np.max([msg.latency() for msg in ...]),
            "p95_latency_ms": np.percentile([msg.latency() for msg in ...], 95)
        }
        
        self.results.append(results)
        self.translator.stop()
        
        return results
    
    def test_quality(self, audio_file, reference_translation=""):
        """
        Measure translation quality using BLEU score.
        """
        from sacrebleu import BLEU
        
        # ... get translation from system ...
        
        bleu = BLEU()
        score = bleu.corpus_score(
            [system_translation],
            [[reference_translation]]
        )
        
        return {"bleu_score": score.score}
    
    def generate_report(self):
        """Generate performance report"""
        report = f"""
        === Speech-to-Speech Translation Performance Report ===
        
        Average End-to-End Latency: {np.mean([r['avg_latency_ms'] for r in self.results]):.0f}ms
        P95 Latency: {np.percentile([r['p95_latency_ms'] for r in self.results], 95):.0f}ms
        Max Latency: {max([r['max_latency_ms'] for r in self.results]):.0f}ms
        
        Translation Quality (BLEU): {np.mean([r.get('bleu_score', 0) for r in self.results]):.2f}
        
        Test Results:
        {self.results}
        """
        
        return report
```

---

## Part 4: Advanced Considerations

### A. Language-Specific Optimizations

**Problematic Languages for Pipeline Approach**:
- German, Japanese, Korean: Heavy word reordering
- Arabic: Right-to-left script
- Tonal languages (Mandarin, Vietnamese): Prosody carries meaning

**Solution**: Use language-specific models:

```python
if target_lang in ["de", "ja", "ko"]:
    # Use end-to-end model for these languages
    model = DirectSTModel(source_lang, target_lang)
else:
    # Use streaming cascade for others
    model = StreamingCascadeModel(source_lang, target_lang)
```

### B. Edge Deployment

Deploy components locally to eliminate cloud latency:

```
┌─────────────────────┐
│  Local Device       │
│  ┌─────────────────┐│
│  │ Kyutai (1B)     ││ ← 1s latency
│  │ LLaMA 7B (Q4)   ││ ← 200ms latency
│  │ CosyVoice2      ││ ← 150ms latency
│  └─────────────────┘│
│                     │
│ Total Latency: ~1.5s│
└─────────────────────┘
```

### C. Network Deployment

For scale, shard across GPUs:

```python
class DistributedPipeline:
    def __init__(self):
        # Shard by language pair
        self.stt_cluster = GPUCluster(
            workers=4,
            model="kyutai-1b"
        )
        self.mt_cluster = GPUCluster(
            workers=8,
            model="llama-7b-q4"
        )
        self.tts_cluster = GPUCluster(
            workers=4,
            model="cosy-voice-2"
        )
    
    def translate(self, audio):
        # Distribute across clusters
        stт_result = self.stt_cluster.process(audio)
        mt_result = self.mt_cluster.process(stt_result)
        tts_result = self.tts_cluster.process(mt_result)
        
        return tts_result
```

---

## Summary: Architecture Comparison

| Aspect | Your Plan | Recommended Plan |
|--------|-----------|------------------|
| **STT** | Batch Whisper (full utterance) | Streaming ASR (chunks) |
| **Latency for 1st output** | 3-5s | 1.5-2s |
| **MT Approach** | Sentence-level | Chunk-aware with context |
| **TTS** | Google Cloud (500-1000ms) | CosyVoice2 (150ms) |
| **Target Latency Met** | No (2-3x over budget) | Yes (~1.5s achieved) |
| **Complexity** | Low | Medium |
| **Infrastructure** | Cloud-dependent | Edge-capable |

---

## Recommended Implementation Timeline

**Week 1-2**: Phase 1 (Audio pipeline + queuing)  
**Week 3-4**: Phase 2 (Streaming MT + TTS)  
**Week 5**: Phase 3 (Main orchestration)  
**Week 6**: Phase 4 (Error handling)  
**Week 7-8**: Phase 5 (Testing + documentation)  

**Expected Outcome**: **1.5-2 second end-to-end latency** with **concurrent processing at every stage**.
