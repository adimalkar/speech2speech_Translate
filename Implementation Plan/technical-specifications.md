# Technical Specifications: Streaming Concurrent S2ST Architecture

## System Requirements

### Functional Requirements

| Requirement | Specification | Priority |
|------------|---------------|----------|
| **Input Format** | PCM 16-bit, 16 kHz, mono audio stream | Critical |
| **Output Format** | PCM 16-bit, 22.05 kHz (TTS standard) audio stream | Critical |
| **Latency (First Output)** | <2 seconds from speech start | Critical |
| **Latency (End-to-End)** | <1.5 seconds for typical 5-second utterance | Critical |
| **Supported Language Pairs** | English ↔ Spanish, French, German, Mandarin | Important |
| **Translation Quality (BLEU)** | ≥25 BLEU score | Important |
| **Speech Quality (MOS)** | ≥4.5 Mean Opinion Score | Important |
| **Error Recovery** | Automatic fallback if any component fails | Important |
| **Concurrent Users** | ≥10 bidirectional conversations (single GPU) | Desirable |

### Non-Functional Requirements

| Requirement | Specification | Rationale |
|------------|---------------|-----------|
| **Availability** | 99.5% uptime | Production requirement |
| **Scalability** | Horizontal scaling via load balancing | Multi-user support |
| **Edge Deployment** | Must run on CPU/lightweight GPU | Offline capability |
| **Memory Usage** | <8GB per concurrent stream | Resource efficiency |
| **Power Consumption** | <50W per stream (edge device) | Portable deployment |
| **Monitoring** | Real-time latency and quality metrics | Operational visibility |

---

## Data Flow Specification

### Audio Chunk Format

```python
# Audio chunk structure flowing through system
class AudioChunk:
    chunk_id: int                    # Unique identifier for tracking
    audio_data: np.ndarray          # PCM samples (float32, normalized -1.0 to 1.0)
    timestamp: float                # Unix timestamp when chunk captured
    duration_ms: int                # Duration of chunk in milliseconds
    sample_rate: int                # Always 16000 Hz
    vad_confidence: float           # 0.0-1.0, confidence speech is present
    is_final: bool                  # True if VAD detected end of utterance
    
# Typical values:
# - duration_ms: 500-1000 (overlapping chunks)
# - audio_data shape: (8000-16000,) for 500-1000ms at 16kHz
```

### Text Token Format

```python
class TextToken:
    chunk_id: int                   # Reference to audio chunk
    text: str                       # Transcribed/translated text
    confidence: float               # 0.0-1.0 confidence in transcription
    token_type: str                 # "partial" or "final"
    source_language: str            # Language code (e.g., "en")
    target_language: str            # For translation stage
    metadata: dict                  # Extra info (context, stability, etc.)
    timestamp: float                # Server timestamp
    processing_latency_ms: float    # Time spent in this stage
```

### Speech Output Format

```python
class SpeechOutput:
    chunk_id: int                   # Trace back to original audio
    audio: np.ndarray               # PCM samples (float32, normalized)
    sample_rate: int                # 22050 Hz (TTS standard)
    duration_ms: int                # Duration of audio
    speaker_id: int                 # Which speaker/voice
    language: str                   # Output language code
    confidence: float               # Overall translation confidence
    total_latency_ms: float         # End-to-end latency for this chunk
```

---

## Component Interface Specifications

### 1. Streaming ASR Interface

```python
class StreamingASRInterface:
    """
    Interface that any streaming ASR model must implement.
    """
    
    def __init__(
        self,
        model_name: str,
        language: str,
        sample_rate: int = 16000
    ):
        """Initialize ASR model"""
        pass
    
    def process_chunk(
        self,
        audio: np.ndarray  # Shape: (N,) where N = sample_rate * duration_s
    ) -> TextToken:
        """
        Process audio chunk and return partial transcription.
        
        Called frequently (every 500ms) with streaming audio.
        
        Args:
            audio: Audio samples (float32, normalized -1 to 1)
        
        Returns:
            TextToken with partial transcription
            
        Latency SLA: <500ms for 500ms audio chunk
        """
        pass
    
    def finalize(self) -> TextToken:
        """
        Called when speech ends (VAD detects silence).
        Returns final, corrected transcription.
        
        Latency SLA: <200ms
        """
        pass
    
    def get_latency_stats(self) -> dict:
        """Return p50, p95, p99 latency metrics"""
        pass
```

### 2. Machine Translation Interface

```python
class StreamingMTInterface:
    """
    Interface for chunk-aware machine translation.
    """
    
    def __init__(
        self,
        model_name: str,
        source_lang: str,
        target_lang: str,
        context_window_size: int = 5
    ):
        """Initialize MT model"""
        pass
    
    def translate_chunk(
        self,
        text: str,
        is_final: bool = False
    ) -> TextToken:
        """
        Translate single chunk with context awareness.
        
        Args:
            text: Text to translate
            is_final: Whether this is final text (affects buffering)
        
        Returns:
            TextToken with translated text
            
        Latency SLA: 
            - Partial: <400ms
            - Final: <500ms
        """
        pass
    
    def set_context(self, context: List[str]):
        """Set previous chunks for context"""
        pass
    
    def clear_context(self):
        """Clear context (e.g., between utterances)"""
        pass
```

### 3. Streaming TTS Interface

```python
class StreamingTTSInterface:
    """
    Interface for low-latency speech synthesis.
    """
    
    def __init__(
        self,
        model_name: str,
        language: str,
        speaker_id: int = 0,
        sample_rate: int = 22050
    ):
        """Initialize TTS model"""
        pass
    
    def synthesize_streaming(
        self,
        text: str,
        is_final: bool = False
    ) -> SpeechOutput:
        """
        Synthesize speech from text with streaming capability.
        
        Args:
            text: Text to synthesize
            is_final: Whether to finalize synthesis
        
        Returns:
            SpeechOutput with audio samples
            
        Latency SLA: <200ms (including inference + encoding)
        """
        pass
    
    def get_available_voices(self) -> List[dict]:
        """Return available speaker configurations"""
        pass
```

### 4. VAD Interface

```python
class RealTimeVADInterface:
    """
    Voice Activity Detection for stream segmentation.
    """
    
    def process_chunk(
        self,
        audio: np.ndarray,
        frame_duration_ms: int = 20
    ) -> Tuple[bool, float]:
        """
        Check if audio chunk contains speech.
        
        Returns:
            (is_speech: bool, confidence: float 0-1)
            
        Latency SLA: <5ms
        """
        pass
    
    def has_speech_ended(
        self,
        silence_duration_ms: int = 400
    ) -> bool:
        """
        Check if enough silence detected to mark end of utterance.
        
        Returns:
            True if utterance end detected
        """
        pass
    
    def reset(self):
        """Reset VAD state for next utterance"""
        pass
```

---

## Queue Message Specification

### Audio Queue

```python
"""
Queue: audio_queue
Purpose: Transport audio chunks from capture to STT
Max size: 20 messages (10 seconds of 500ms chunks)

Message format: Tuple[str, np.ndarray, dict]
- Message type: "audio_chunk"
- Audio data: numpy array
- Metadata: timestamps, indices
"""

Message = ("audio_chunk", audio_array, {
    "chunk_id": 1,
    "timestamp": 1699255200.123,
    "vad_confidence": 0.95,
    "is_final": False
})
```

### Partial Text Queue

```python
"""
Queue: partial_text_queue  
Purpose: Transport partial transcriptions from STT to MT
Max size: 20 messages

Message format: TextToken objects
"""

Message = TextToken(
    chunk_id=1,
    text="Hello, how are",
    confidence=0.92,
    token_type="partial",
    source_language="en",
    timestamp=1699255200.150,
    processing_latency_ms=50
)
```

### Translated Queue

```python
"""
Queue: translated_queue
Purpose: Transport translated text from MT to TTS
Max size: 20 messages

Message format: TextToken objects with target_language set
"""

Message = TextToken(
    chunk_id=1,
    text="Hola, ¿cómo estás?",
    confidence=0.88,
    token_type="partial",
    source_language="en",
    target_language="es",
    timestamp=1699255200.200,
    processing_latency_ms=75
)
```

### Speech Output Queue

```python
"""
Queue: speech_output_queue
Purpose: Transport synthesized audio to playback
Max size: 10 messages (2-3 seconds of audio)

Message format: SpeechOutput objects
"""

Message = SpeechOutput(
    chunk_id=1,
    audio=audio_array,
    sample_rate=22050,
    duration_ms=500,
    speaker_id=0,
    language="es",
    confidence=0.88,
    total_latency_ms=350
)
```

---

## Thread Synchronization Specification

### Thread Safety Requirements

```python
# All queue operations must be thread-safe
# Use Python's thread.Lock() for:
# 1. Metrics aggregation
# 2. Model state updates (if not inherently thread-safe)
# 3. File I/O (logging)

class ThreadSafeMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.latencies = {"stt": [], "mt": [], "tts": []}
    
    def record_latency(self, stage: str, latency_ms: float):
        with self._lock:
            self.latencies[stage].append(latency_ms)
```

### Deadlock Prevention

```python
# Queue ordering (prevent circular dependencies):
# 1. Audio Queue (source)
# 2. STT → Partial Text Queue
# 3. Partial Text Queue → MT
# 4. Translated Queue
# 5. TTS → Speech Output Queue
# 6. Speech Output Queue (sink)

# NEVER have circular dependencies between threads
# NEVER hold locks while waiting for queue operations
```

---

## Error Handling Specification

### Component Failure Modes

| Component | Failure Mode | Recovery |
|-----------|-------------|----------|
| **Streaming STT** | STT model crashes | Fall back to buffering + Whisper API |
| **MT Model** | OOM or slow inference | Queue MT requests, use fallback API |
| **TTS** | Synthesis fails | Use alternative TTS (EdgeTTS, then Google) |
| **Audio Input** | Microphone disconnected | Log error, pause system, retry on reconnect |
| **Network** | API timeouts | Queued retry with exponential backoff |

### Implementation Example

```python
class RobustTranslator:
    def __init__(self):
        self.primary_stt = StreamingASR("kyutai")
        self.fallback_stt = WhisperAPI()  # Cloud fallback
        
        self.primary_mt = LLaMAMT()
        self.fallback_mt = GoogleTranslateAPI()
        
        self.primary_tts = CosyVoice2()
        self.fallback_tts = EdgeTTS()
    
    def translate_with_fallback(self, audio: np.ndarray) -> SpeechOutput:
        """
        Translate with automatic fallback at each stage.
        """
        try:
            # Try primary STT
            text = self.primary_stt.process_chunk(audio)
        except:
            # Fall back to cloud STT
            text = self.fallback_stt.transcribe(audio)
        
        try:
            # Try primary MT
            translated = self.primary_mt.translate_chunk(text)
        except:
            # Fall back to Google Translate
            translated = self.fallback_mt.translate(text)
        
        try:
            # Try primary TTS
            audio_out = self.primary_tts.synthesize_streaming(translated)
        except:
            # Fall back to Edge TTS
            audio_out = self.fallback_tts.synthesize(translated)
        
        return audio_out
```

---

## Performance Specification

### Latency Budget Allocation (for 1.5s target)

```
Total budget: 1500 ms

Breakdown:
- Chunk aggregation (VAD): 100 ms (start early)
- Streaming STT first chunk: 800 ms (start here)
- Streaming MT first chunk: 300 ms (overlaps with STT)
- Streaming TTS first chunk: 150 ms (overlaps with MT)
- Audio playback buffer: 50 ms (network/scheduling)
- Margin: 100 ms
─────────────────────────────
Total: 1500 ms ✓

Note: Stages overlap due to concurrent processing
```

### Per-Stage SLA

| Stage | Latency SLA | P95 | P99 |
|-------|-------------|-----|-----|
| **VAD Processing** | <5ms | <10ms | <20ms |
| **STT (per chunk)** | <500ms | <800ms | <1000ms |
| **MT (per chunk)** | <400ms | <500ms | <700ms |
| **TTS (per chunk)** | <200ms | <250ms | <350ms |

### Throughput Specification

```python
# Single GPU (NVIDIA T4)
stt_throughput = 4-8 concurrent streams
mt_throughput = 16-32 concurrent streams (lighter)
tts_throughput = 8-16 concurrent streams

# With GPU optimization
total_throughput = ~20 concurrent bidirectional conversations

# Scaling calculation:
# For 1M concurrent users → need ~50,000 GPUs
# Cost: $25k/hour = $18M/month (unoptimized)
# With batching/optimization: ~$2M/month
```

---

## Monitoring and Observability

### Metrics to Collect

```python
class MetricsCollector:
    def __init__(self):
        # Latency metrics (in milliseconds)
        self.latencies = {
            "stt_per_chunk": [],
            "mt_per_chunk": [],
            "tts_per_chunk": [],
            "e2e_per_utterance": [],  # End-to-end
            "first_output_latency": []  # Time to first output
        }
        
        # Quality metrics
        self.quality = {
            "stt_wer": [],  # Word Error Rate
            "mt_bleu": [],  # BLEU score
            "tts_mos": [],  # Mean Opinion Score
        }
        
        # Reliability metrics
        self.reliability = {
            "fallback_count": 0,
            "error_count": 0,
            "uptime_pct": 99.5
        }
        
        # Throughput metrics
        self.throughput = {
            "chunks_processed": 0,
            "utterances_completed": 0,
            "avg_throughput_per_sec": 0
        }

# Logging format (structured JSON)
logging_format = {
    "timestamp": "2024-01-15T10:30:45.123Z",
    "event": "chunk_processed",
    "chunk_id": 1,
    "stage": "stt",
    "latency_ms": 450,
    "confidence": 0.92,
    "text_length": 12,
    "error": None  # Or error message if failed
}
```

### Dashboard Metrics

```
Real-time Dashboard:
┌────────────────────────────────────┐
│ Current Latency: 1.35s ✓           │
│ P95 Latency: 1.52s ✓               │
│ Average BLEU: 28.3                 │
│ Uptime: 99.87%                     │
│ Active Users: 42/100               │
│ Queue Depths: STT=2, MT=3, TTS=1   │
└────────────────────────────────────┘
```

---

## Configuration Parameters

### Tunable Parameters for Latency/Quality Tradeoff

```python
# Audio chunking
CHUNK_DURATION_MS = 500      # Smaller = lower latency, more context switching
CHUNK_OVERLAP_MS = 100       # For smoother transitions

# VAD
VAD_SILENCE_DURATION_MS = 400  # Time to wait for utterance end
VAD_FRAME_DURATION_MS = 20     # Processing window

# STT
STT_BEAM_SIZE = 3            # Smaller = faster, less accurate
STT_EARLY_EMIT_THRESHOLD = 0.7  # Confidence to emit partial results

# MT  
MT_CONTEXT_WINDOW_SIZE = 5   # Chunks to remember for context
MT_AGREEMENT_THRESHOLD = 0.8 # Stability threshold for emission

# TTS
TTS_STREAMING_MODE = True    # Enable chunk-by-chunk synthesis
TTS_VOICE_ID = 0             # Speaker selection

# System
MAX_QUEUE_SIZE = 20          # Queue depth
MONITORING_INTERVAL_SEC = 5  # Metrics collection frequency
FALLBACK_TIMEOUT_SEC = 2     # Max wait before using fallback
```

---

## Deployment Configurations

### Configuration 1: Local Development (CPU)

```python
config_dev = {
    "stt_model": "kyutai-1b",      # Lightweight
    "stt_device": "cpu",           # CPU inference
    "mt_model": "mistral-7b-q3",   # Quantized
    "mt_device": "cpu",
    "tts_model": "cosy-voice-2-0.5b",
    "tts_device": "cpu",
    "chunk_size_ms": 500,
    "expected_latency_ms": 3500-4000  # Slower but works
}
```

### Configuration 2: Edge Deployment (Jetson)

```python
config_edge = {
    "stt_model": "kyutai-1b",
    "stt_device": "gpu",  # Jetson GPU
    "mt_model": "phi-2-q4",  # Tiny LLM
    "mt_device": "gpu",
    "tts_model": "cosy-voice-2-0.5b",
    "tts_device": "gpu",
    "chunk_size_ms": 500,
    "expected_latency_ms": 2000-2500
}
```

### Configuration 3: Cloud Deployment (Multi-GPU)

```python
config_cloud = {
    "stt_cluster": {
        "model": "kyutai-1b",
        "device": "gpu",
        "replicas": 4,      # Load balanced
        "batch_size": 8
    },
    "mt_cluster": {
        "model": "llama-7b-q4",
        "device": "gpu",
        "replicas": 8,
        "batch_size": 16
    },
    "tts_cluster": {
        "model": "cosy-voice-2-0.5b",
        "device": "gpu",
        "replicas": 4,
        "batch_size": 8
    },
    "expected_latency_ms": 1200-1500
}
```

---

## Validation Checklist

Before deployment, verify:

- [ ] STT latency <500ms per 500ms chunk
- [ ] MT latency <400ms per chunk
- [ ] TTS latency <200ms per chunk
- [ ] End-to-end latency <2s for first output
- [ ] Translation BLEU score >25 on test set
- [ ] Speech quality MOS >4.5
- [ ] No memory leaks after 24-hour run
- [ ] Graceful fallback to secondary models works
- [ ] Monitoring dashboard shows all metrics
- [ ] Load test passes (10+ concurrent streams)
- [ ] Error recovery verified
- [ ] Documentation complete
