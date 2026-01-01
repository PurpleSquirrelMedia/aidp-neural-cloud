# GPU Integration Documentation

## How AIDP Neural Cloud Uses GPU Compute

This document explains how Neural Cloud leverages AIDP's decentralized GPU network for LLM inference.

---

## GPU Utilization Overview

### 1. Model Loading (VRAM)

Models are loaded into GPU VRAM with optimized quantization:

| Model | Full Precision | 4-bit Quantized | VRAM Savings |
|-------|----------------|-----------------|--------------|
| 7B params | 28 GB | ~5 GB | 82% |
| 8B params | 32 GB | ~6 GB | 81% |
| 13B params | 52 GB | ~9 GB | 83% |
| 70B params | 280 GB | ~40 GB | 86% |

**Quantization method**: NF4 (4-bit NormalFloat) with double quantization

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
```

### 2. Inference (CUDA Cores + Tensor Cores)

LLM inference leverages multiple GPU components:

| Component | GPU Hardware | Operation |
|-----------|-------------|-----------|
| Attention | Tensor Cores | Matrix multiplication |
| FFN layers | CUDA Cores | Element-wise ops |
| KV Cache | VRAM | Attention caching |
| Sampling | CUDA Cores | Top-p, temperature |

### 3. Continuous Batching

Neural Cloud uses continuous batching for high throughput:

```
Traditional Batching:
Request 1: [====]
Request 2:       [======]
Request 3:             [===]
           ↑ GPU idle time

Continuous Batching:
Request 1: [====]
Request 2: [======]
Request 3:   [===]
           ↑ GPU fully utilized
```

---

## AIDP Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Client Request                            │
│  POST /v1/chat/completions                                  │
│  {"model": "purple-squirrel-r1", "messages": [...]}        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Neural Cloud API                          │
│  1. Validate request                                        │
│  2. Format prompt for model                                 │
│  3. Query load balancer                                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                            │
│  Routing Strategy:                                          │
│  1. Filter by model availability                            │
│  2. Filter by health status                                 │
│  3. Score: latency(40%) + load(40%) + reliability(20%)     │
│  4. Route to best node                                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   AIDP GPU Node                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  NVIDIA GPU (A100/A10G/RTX 4090)                    │   │
│  │                                                      │   │
│  │  VRAM Layout:                                        │   │
│  │  ├── Model weights (4-bit): 5-6 GB                  │   │
│  │  ├── KV Cache: 1-4 GB (dynamic)                     │   │
│  │  ├── Activations: 0.5-1 GB                          │   │
│  │  └── CUDA context: 0.5 GB                           │   │
│  │                                                      │   │
│  │  Inference Engine: vLLM / Transformers              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Pipeline:                                                  │
│  prompt → tokenize → prefill → decode → detokenize         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              AIDP Proof & Verification                      │
│  1. Hash of input/output                                    │
│  2. GPU utilization metrics                                 │
│  3. Verifiable compute proof                                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Response                                 │
│  {"choices": [{"message": {"content": "..."}}]}            │
└─────────────────────────────────────────────────────────────┘
```

---

## GPU Requirements by Model

| Model | Min VRAM | Recommended GPU | Tokens/sec |
|-------|----------|-----------------|------------|
| phi-3-mini (3.8B) | 3 GB | RTX 3060 | ~80 |
| mistral-7b | 5 GB | RTX 3070 | ~45 |
| qwen2-7b | 5 GB | RTX 3070 | ~45 |
| deepseek-coder-6.7b | 5 GB | RTX 3070 | ~50 |
| llama-3.1-8b | 6 GB | RTX 3080 | ~40 |
| purple-squirrel-r1 (8B) | 6 GB | RTX 3080 | ~40 |

---

## Performance Benchmarks

### Test Configuration
- Model: Purple Squirrel R1 (8B, 4-bit)
- GPU: NVIDIA A10G (AIDP node)
- Prompt: 100 tokens average
- Generation: 256 tokens

### Latency Results

| Metric | Value |
|--------|-------|
| Time to First Token (TTFT) | 85ms |
| Inter-Token Latency (ITL) | 22ms |
| Total generation (256 tokens) | 5.7s |
| Throughput | 45 tokens/sec |

### Batch Inference

| Batch Size | Total Tokens | Time | Throughput |
|------------|--------------|------|------------|
| 1 | 256 | 5.7s | 45 tok/s |
| 4 | 1024 | 8.2s | 125 tok/s |
| 8 | 2048 | 12.1s | 169 tok/s |
| 16 | 4096 | 20.5s | 200 tok/s |

### Cost Comparison

| Provider | Cost per 1M tokens |
|----------|-------------------|
| OpenAI GPT-4o-mini | $0.15 |
| Anthropic Claude Haiku | $0.25 |
| Together.ai Llama | $0.20 |
| **AIDP Neural Cloud** | **$0.08** |

**AIDP provides 47-68% cost savings vs. centralized providers.**

---

## Code Examples

### OpenAI-Compatible Client
```python
import openai

client = openai.OpenAI(
    base_url="https://neural-cloud.aidp.store/v1",
    api_key="your-aidp-api-key"
)

response = client.chat.completions.create(
    model="purple-squirrel-r1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is AIDP?"}
    ],
    max_tokens=256,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Streaming Response
```python
stream = client.chat.completions.create(
    model="purple-squirrel-r1",
    messages=[{"role": "user", "content": "Explain GPU compute"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Batch Inference
```python
import requests

response = requests.post(
    "https://neural-cloud.aidp.store/batch",
    headers={"Authorization": "Bearer your-api-key"},
    json={
        "prompts": [
            "What is blockchain?",
            "Explain GPU acceleration",
            "What is decentralized compute?"
        ],
        "model": "purple-squirrel-r1",
        "max_tokens": 256
    }
)

for result in response.json()["results"]:
    print(f"Q: {result['prompt']}")
    print(f"A: {result['response']}\n")
```

---

## Multi-Node Distribution

Neural Cloud distributes load across multiple AIDP GPU nodes:

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │  Node 1  │      │  Node 2  │      │  Node 3  │
    │  A10G    │      │  A100    │      │  RTX4090 │
    │  Load:20%│      │  Load:45%│      │  Load:30%│
    └──────────┘      └──────────┘      └──────────┘
```

**Benefits**:
- No single point of failure
- Geographic distribution for lower latency
- Automatic failover on node failure
- Scale to thousands of concurrent requests

---

## AIDP Token Integration

Neural Cloud supports payment with AIDP tokens:

- **Token**: AIDP (Solana SPL)
- **Contract**: `PLNk8NUTBeptajEX9GzZrxsYPJ1psnw62dPnWkGcyai`
- **Pricing**: ~$0.08 per 1M tokens (varies by node)

---

## Links

- [AIDP Network](https://aidp.store)
- [vLLM Documentation](https://docs.vllm.ai)
- [Transformers Quantization](https://huggingface.co/docs/transformers/quantization)
- [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
