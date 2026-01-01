# AIDP Neural Cloud

Distributed LLM Inference on AIDP Decentralized GPU Network

## Overview

Neural Cloud provides OpenAI-compatible LLM inference powered by AIDP's decentralized GPU network:
- **Low latency inference** across distributed GPU nodes
- **Cost-efficient** compared to centralized providers
- **Fault tolerant** with automatic failover
- **Scalable** to thousands of concurrent requests

## GPU Usage on AIDP

This project demonstrates intensive GPU compute on AIDP:

| Component | GPU Utilization | Description |
|-----------|-----------------|-------------|
| Model Loading | VRAM | 4-bit quantized model in GPU memory |
| Inference | CUDA cores | Transformer forward pass |
| KV Cache | VRAM | Attention key-value caching |
| Batching | Tensor cores | Dynamic request batching |

### Models Deployed

| Model | Parameters | Quantization | VRAM Required |
|-------|------------|--------------|---------------|
| Purple Squirrel R1 | 8B | 4-bit NF4 | ~6GB |
| Llama 3.1 8B | 8B | 4-bit | ~6GB |
| Mistral 7B | 7B | 4-bit | ~5GB |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Neural Cloud                           │
├─────────────────────────────────────────────────────────┤
│  API Gateway                                            │
│  └── /v1/chat/completions (OpenAI-compatible)          │
│  └── /v1/models, /health, /metrics                     │
├─────────────────────────────────────────────────────────┤
│  Load Balancer                                          │
│  └── Health checks → Route to fastest node             │
├─────────────────────────────────────────────────────────┤
│  AIDP GPU Workers (N nodes)                            │
│  └── vLLM inference engine                             │
│  └── Continuous batching                               │
│  └── PagedAttention for KV cache                       │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure AIDP credentials
export AIDP_API_KEY="your-api-key"
export AIDP_WALLET="your-solana-wallet"

# Start the API server
python src/api.py

# Test inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "purple-squirrel-r1",
    "messages": [{"role": "user", "content": "What is AIDP?"}]
  }'
```

## OpenAI-Compatible API

### Chat Completions
```python
import openai

client = openai.OpenAI(
    base_url="https://neural-cloud.aidp.store/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="purple-squirrel-r1",
    messages=[
        {"role": "user", "content": "Explain decentralized GPU compute"}
    ]
)
print(response.choices[0].message.content)
```

### Batch Inference
```python
# Process multiple prompts efficiently
response = requests.post(
    "https://neural-cloud.aidp.store/batch",
    json={
        "prompts": [
            "What is blockchain?",
            "Explain GPU acceleration",
            "What is AIDP?"
        ],
        "max_tokens": 256
    }
)
```

## Project Structure

```
aidp-neural-cloud/
├── src/
│   ├── api.py            # FastAPI server
│   ├── aidp_client.py    # AIDP GPU integration
│   ├── inference.py      # vLLM inference engine
│   ├── load_balancer.py  # Node routing
│   └── models.py         # Model configurations
├── api/
│   └── openai_compat.py  # OpenAI compatibility layer
├── docker/
│   └── Dockerfile.gpu    # CUDA Docker image
├── tests/
│   └── test_api.py       # API tests
└── docs/
    └── BENCHMARKS.md     # Performance comparisons
```

## AIDP Integration

### How We Use AIDP GPUs

1. **Model Deployment**: Models loaded into VRAM on AIDP GPU nodes
2. **Request Routing**: API gateway routes to healthy nodes
3. **GPU Inference**: vLLM executes inference with CUDA
4. **Verification**: AIDP proofs verify computation
5. **Response**: Results returned with usage metrics

### Distributed Benefits

- **Geographic distribution**: Lower latency globally
- **Redundancy**: No single point of failure
- **Cost**: 50-70% cheaper than OpenAI/Anthropic APIs
- **Privacy**: Requests processed on decentralized nodes

## Benchmarks

| Metric | Neural Cloud (AIDP) | OpenAI GPT-4o-mini | Savings |
|--------|---------------------|---------------------|---------|
| Latency (p50) | 180ms | 250ms | 28% faster |
| Cost per 1M tokens | $0.08 | $0.15 | 47% cheaper |
| Throughput | 50 req/s | N/A | Scalable |

## Demo Video

[Watch the 2-minute demo](./docs/demo.mp4) showing:
1. API request/response flow
2. Multi-node load balancing
3. Latency comparison
4. Cost breakdown

## Links

- [AIDP Marketplace Listing](https://aidp.store/marketplace/neural-cloud)
- [Twitter/X](https://x.com/purplesquirrelnetworks)
- [API Docs](https://neural-cloud.aidp.store/docs)

## License

MIT License - Built for AIDP GPU Build & Recruit Campaign
