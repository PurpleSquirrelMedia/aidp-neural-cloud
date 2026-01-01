# AIDP Hackathon Submission Checklist

## Project: AIDP Neural Cloud
**Category**: Best Submission or Recruited Project

---

## Required Submissions

### 1. AIDP Marketplace Project Page
- [ ] Create account on https://aidp.store
- [ ] Navigate to marketplace submission
- [ ] Fill in project details:
  - **Name**: AIDP Neural Cloud
  - **Description**: Distributed LLM inference on decentralized GPUs
  - **Category**: AI / LLM Inference
  - **GPU Usage**: Model loading, CUDA inference, KV cache

### 2. Public GitHub Repository
- [x] Repository created: https://github.com/PurpleSquirrelMedia/aidp-neural-cloud
- [x] Code is public
- [x] README with clear documentation
- [x] GPU integration documentation
- [x] Working code examples

### 3. Social Media Link (Twitter/X)
- [ ] Post announcement using template from `docs/SOCIAL_POST.md`
- [ ] Include:
  - Project name and description
  - @aidpstore mention
  - GitHub link
  - Demo video link (if available)
  - Relevant hashtags (#AIDP #AI #LLM #DePIN)

### 4. Demo Video (1-2 minutes)
- [ ] Record using script from `docs/SOCIAL_POST.md`
- [ ] Show:
  - API request in terminal
  - Streaming response
  - OpenAI compatibility demo
  - Multi-model support
  - Cost comparison
- [ ] Upload to YouTube/Loom/etc.
- [ ] Include link in marketplace submission

### 5. GPU Usage Explanation
- [x] Documented in `docs/GPU_INTEGRATION.md`
- [x] Covers:
  - 4-bit quantization on GPU
  - CUDA inference execution
  - KV cache management
  - Continuous batching
  - Performance benchmarks

---

## Judging Criteria Alignment

| Criteria | How We Address It |
|----------|-------------------|
| Technical execution | Full OpenAI-compatible API with load balancing |
| GPU integration depth | Quantized model loading, CUDA inference, KV cache |
| Product quality | 6 models, streaming, batch inference |
| Creativity & originality | Decentralized LLM API platform |
| User experience & design | Drop-in OpenAI replacement |
| Vision & scalability | Multi-node distribution, auto-failover |
| Social proof | Open source, Purple Squirrel R1 model |
| Depth of AIDP compute usage | Continuous GPU utilization for inference |
| Value added to ecosystem | Enables AI apps to use AIDP |

---

## Quick Links

- **GitHub**: https://github.com/PurpleSquirrelMedia/aidp-neural-cloud
- **AIDP**: https://aidp.store
- **Demo Script**: `python scripts/demo-inference.py`

---

## Submission Commands

```bash
# Test locally before submission
cd ~/Projects/aidp-neural-cloud
pip install -r requirements.txt
python scripts/demo-inference.py

# Start API server (local testing)
uvicorn src.api:app --reload

# Test with curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "purple-squirrel-r1", "messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## After Submission

- [ ] Monitor for questions/feedback
- [ ] Engage with AIDP community on Telegram
- [ ] Share progress updates on Twitter
- [ ] Prepare for demo day (if applicable)
