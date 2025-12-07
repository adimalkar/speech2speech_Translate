# Quick Reference: Key Implementation Changes

## Your Current Plan vs. Recommended Plan

### Component Selection Changes

#### Speech-to-Text (STT)
| Aspect | Current | Recommended |
|--------|---------|-------------|
| Model | OpenAI Whisper (base) | Kyutai 1B or RNN-T models |
| Processing | Batch (wait for silence) | Streaming (chunk-by-chunk) |
| Latency | 1500-2000ms | 1000-1200ms (first token) |
| Output | Complete transcription | Partial + streaming updates |
| Accuracy (WER) | 4-5% | 6.4% (acceptable for streaming) |

#### Machine Translation (MT)
| Aspect | Current | Recommended |
|--------|---------|-------------|
| Approach | Google Cloud Translation API | LLM-based (LLaMA 7B quantized) |
| Processing | Full sentence at once | Chunk-aware with context |
| Latency | 300-500ms | 200-400ms per chunk |
| Quality | High (commercial API) | High (LLM with context) |
| Reordering Support | Limited | Excellent |

#### Text-to-Speech (TTS)
| Aspect | Current | Recommended |
|--------|---------|-------------|
| Model | Google Cloud TTS | CosyVoice2 (0.5B) or EdgeTTS |
| Latency | 500-1000ms | **150ms** (critical improvement) |
| Streaming | No | Yes (chunk-based) |
| Quality | Excellent | Excellent (5.53 MOS score) |
| Deployment | Cloud-only | Edge-capable |

### Architecture Changes

#### Threading Model
```
CURRENT (3 threads):
┌─ Audio Capture (input only)
├─ Translation (STT + MT)
└─ TTS (output only)

RECOMMENDED (4 threads + specialized workers):
┌─ Audio Capture & VAD (chunk emission)
├─ Streaming STT (partial transcription)
├─ Streaming MT (chunk translation)
├─ Streaming TTS (synthesis)
└─ Output Handler (playback)
```

#### Queue Architecture
```
CURRENT (2 queues):
Audio Queue → [Translation] → Translation Queue → [TTS] → Output

RECOMMENDED (4+ queues):
Audio Queue → [STT] → Partial Text Queue → [MT] → Translated Queue → [TTS] → Speech Queue → [Output]
```

### Performance Targets

| Metric | Current Plan | Recommended | Industry Benchmark |
|--------|-------------|-------------|-------------------|
| **End-to-End Latency** | 3.3-4.5s ❌ | **1.5-2s** ✅ | <1.5s (real interpreters) |
| **First Output Latency** | 3-5s | **1.2-1.5s** ✅ | <1s (ideal) |
| **Translation Quality (BLEU)** | 30-35 | 28-32 | 25-40 range |
| **Speech Quality (MOS)** | 4.2-4.5 | 4.8-5.1 | >4.5 acceptable |
| **Throughput (concurrent streams)** | 50-100 | 200-400+ | Depends on hardware |

---

## Critical Implementation Decisions

### 1. Streaming ASR Selection

**Recommended**: Kyutai 1B or 2.6B
- **Why**: Designed explicitly for low-latency streaming (1-2.5s startup)
- **Tradeoff**: 6.4% WER vs. Whisper's 4% (acceptable for real-time)
- **Deployment**: Can run on edge (CPU/GPU efficient)

**Alternative**: Fine-tuned Whisper with streaming wrapper
- Maintains higher accuracy (4-5% WER)
- Requires more latency optimization
- Works well for high-resource languages

### 2. Machine Translation Approach

**Option A: LLM-Based (Recommended for production)**
```python
# Pros:
- Context awareness across chunks
- Better reordering for difficult languages
- Can be fine-tuned for domain-specific
- Emerging end-to-end capabilities

# Cons:
- Requires quantization for speed
- Higher compute cost
- Cold start latency

# Deploy: LLaMA 7B-Q4 or Mistral 7B-Q4
# Latency target: 200-400ms per chunk
```

**Option B: API-Based (Easier, less latency-tuning)**
```python
# Pros:
- Proven reliability
- No infrastructure cost
- Handles error cases

# Cons:
- Network latency (300-500ms minimum)
- Cannot do chunk-aware translation
- Limited context utilization

# Only use if: You want minimal development time
```

**Option C: End-to-End Models (Future-proof)**
```python
# Emerging: Direct Speech → Speech translation
# Models: StreamSpeech, SLAM-TR
# Advantage: Lower error propagation, fastest latency
# Disadvantage: Limited language pair support

# Recommendation: Hybrid approach
- Use end-to-end for major language pairs (En-De, En-Zh)
- Use cascade for rare pairs (En-Xhosa, En-Tagalog)
```

### 3. TTS Technology Selection

**Critical for achieving 1.5s latency target**

| Technology | Latency | Quality | Edge Capable | Recommendation |
|------------|---------|---------|--------------|-----------------|
| **CosyVoice2** | **150ms** | 5.53 MOS | ✅ Yes | **PRIMARY** |
| EdgeTTS | 200-300ms | 4.8 MOS | ✅ Yes | **SECONDARY** |
| Google Cloud TTS | 500-1000ms | 4.9 MOS | ❌ No | Fallback only |
| Microsoft Azure TTS | 400-600ms | 4.8 MOS | ❌ No | Fallback only |

**Selection Logic**:
```python
if latency_budget <= 500ms:
    use CosyVoice2  # Non-negotiable for <1.5s target
elif latency_budget <= 2s:
    use EdgeTTS  # Good balance
else:
    use CloudAPIs  # Maximum quality, accept higher latency
```

### 4. VAD Strategy

**Recommended**: Hybrid approach

```python
# Use Silero VAD for:
- Real-time chunk detection
- Efficient (CPU: <5ms per chunk)
- Low WER in noisy environments

# Configuration:
frame_duration = 20ms  # 20ms frames
silence_duration = 400-600ms  # End of utterance detection
frame_rate = 50 frames/second

# For streaming:
- Don't wait for complete silence before starting translation
- Start translation after first complete semantic unit (~1-2 chunks)
- Use "Agreement Policy" to emit stable translations
```

---

## Implementation Checklist

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up Python environment with required packages
- [ ] Implement `ChunkStreamingAudioCapture` class
- [ ] Implement queue-based message passing system
- [ ] Create metrics collection framework
- [ ] Test audio capture + VAD with sample audio files

### Phase 2: Streaming Components (Weeks 3-4)
- [ ] Integrate Kyutai streaming ASR
- [ ] Integrate LLM-based MT (LLaMA 7B quantized)
- [ ] Implement `AgreementPolicy` for stable translations
- [ ] Test STT + MT latency on recorded audio
- [ ] Optimize quantization for target hardware

### Phase 3: Integration (Week 5-6)
- [ ] Integrate CosyVoice2 streaming TTS
- [ ] Implement all 4 worker threads
- [ ] Test concurrent processing with real audio
- [ ] Measure end-to-end latency
- [ ] Implement fallback mechanisms

### Phase 4: Optimization (Week 7)
- [ ] Profile bottlenecks using `cProfile`
- [ ] Optimize VAD frame size
- [ ] Tune chunk sizes for optimal latency/quality tradeoff
- [ ] Test with multiple language pairs
- [ ] Stress test with concurrent users (if applicable)

### Phase 5: Polish (Week 8)
- [ ] Add error handling and recovery
- [ ] Create comprehensive logging
- [ ] Build simple UI (Streamlit/Gradio)
- [ ] Generate performance report
- [ ] Document deployment instructions

---

## Hardware Recommendations

### Development Machine
```
CPU: 8+ cores (for parallelization)
RAM: 16GB (for model quantization in memory)
GPU: NVIDIA GPU with 6GB+ VRAM (optional but recommended)
  - With GPU: Can run all models locally (~2.5s latency)
  - Without GPU: Can run smaller models (~3-4s latency)
```

### Edge Deployment
```
Device: NVIDIA Jetson Orin Nano / Google Coral
Models: Kyutai 1B + LLaMA 3.2 1B-Instruct + CosyVoice2 0.5B
Expected latency: 2-3 seconds
Power consumption: <15W
```

### Cloud Deployment
```
Instance: NVIDIA A100 or multiple T4s
Model sharding:
  - STT: 1x T4 (batched)
  - MT: 2-4x T4 (load balanced)
  - TTS: 1-2x T4 (batched)
Expected latency: <1.5 seconds
Cost per request: $0.02-0.05
```

---

## Troubleshooting Guide

### Problem: Latency Exceeds 2 Seconds

**Diagnostic Steps**:
```python
# Check which stage is bottleneck
metrics.average_latency("stt")   # Should be <1200ms
metrics.average_latency("mt")    # Should be <400ms
metrics.average_latency("tts")   # Should be <200ms

# Common causes and fixes:
if stt_latency > 1500:
    # Switch to Kyutai or use GPU acceleration
    # Check chunk size (should be 500-1000ms audio)
    
elif mt_latency > 500:
    # Reduce context window size
    # Use quantized model (Q4 instead of FP16)
    # Reduce beam search size
    
elif tts_latency > 300:
    # Verify CosyVoice2 is being used
    # Check GPU memory availability
    # Reduce chunk size for TTS input
```

### Problem: Translation Quality Degrades

**Causes**:
1. Context window too small (increase to 5-10 chunks)
2. Agreement policy too aggressive (lower overlap threshold)
3. MT model poorly quantized (use Q4 or Q5 instead of Q3)

**Solution**:
```python
# Relax agreement policy
agreement_policy = AgreementPolicy(agreement_threshold=0.6)

# Increase context
mt.max_context_chunks = 10  # Instead of 5

# Better quantization
model = quantize(llama_model, type="q5_k_m")
```

### Problem: Memory Leaks in Long Sessions

**Causes**: Queue buffers growing unbounded, model state not cleared

**Solution**:
```python
# Add periodic cleanup
def cleanup_worker():
    while not stop_event.is_set():
        # Clear old messages from queues
        while audio_queue.qsize() > 20:
            audio_queue.get()
        
        # Clear model caches
        torch.cuda.empty_cache()
        
        time.sleep(10)
```

---

## Testing Scenarios

### Test 1: Short Utterances (3-5 seconds)
```
Input: "Hola, ¿cómo estás?"
Expected latency: 1.5-2s
Pass if: Audio output starts within 2s
```

### Test 2: Long Utterances (15-30 seconds)
```
Input: Paragraph-long speech
Expected latency: Incremental output every 1-2s
Pass if: User sees intermediate results
```

### Test 3: Noisy Environment
```
Input: Restaurant background noise + speech
Expected: VAD filters background, maintains accuracy
Pass if: WER <10% despite noise
```

### Test 4: Language Pair with Heavy Reordering
```
Language pair: English → German
Input: "The quick brown fox jumps over the lazy dog"
Expected: Correct reordering in German
Pass if: BLEU score >25
```

### Test 5: Rapid Context Switching
```
Input: User A speaks EN, User B speaks FR, User A speaks EN again
Expected: System correctly maintains language context
Pass if: No language mixing or context corruption
```

---

## Cost-Benefit Analysis

### Your Current Plan (If Fully Implemented)
- Development time: 6-8 weeks
- Operational latency: 3.3-4.5 seconds ❌ (misses target)
- Monthly cloud cost: $500-1000/month (if scaled)
- Infrastructure complexity: Low

### Recommended Plan
- Development time: 8 weeks (slightly longer, more optimized)
- Operational latency: **1.5-2 seconds** ✅ (meets target)
- Monthly cloud cost: $200-500/month (better optimization)
- Infrastructure complexity: Medium
- **Benefit**: 2-3x latency reduction, meets real-time requirements

### ROI Justification
- Achieves project objectives (real-time, low-latency)
- Scales better (edge deployment option)
- Maintains quality while improving speed
- Worth the extra 2 weeks of development

---

## Migration Path (If Starting from Current Plan)

**Week 1**: Keep current implementation, run baseline latency tests
**Week 2**: Swap out Whisper → Kyutai for streaming ASR
**Week 3**: Add streaming MT with context awareness
**Week 4**: Replace Google Cloud TTS → CosyVoice2
**Week 5**: Measure new latency (should be ~1.5-2s)
**Week 6-8**: Polish, optimize, test

This incremental approach lets you validate improvements at each step.

---

## Next Steps

1. **Review this guide** with your project team
2. **Decide on MT approach** (LLM vs. API vs. End-to-End)
3. **Start Phase 1** (audio pipeline implementation)
4. **Test streaming ASR models** (benchmark latency on your hardware)
5. **Document hardware requirements** (needed for team)

**Questions to resolve before starting**:
- [ ] What hardware will you deploy on? (Cloud vs. Edge vs. Local)
- [ ] Which language pairs are priority? (affects model selection)
- [ ] What's your accuracy tolerance? (affects quantization decisions)
- [ ] Do you need bidirectional support initially? (affects architecture)
- [ ] What's your deployment timeline? (affects component selection)
