#!/usr/bin/env python3
"""
AIDP Neural Cloud - Demo Script
Demonstrates OpenAI-compatible LLM inference
"""

import asyncio
import time
import json

# Simulated responses for demo (replace with real AIDP calls)
DEMO_RESPONSES = {
    "What is AIDP?": """AIDP (AI Decentralized Platform) is a decentralized GPU compute network that powers global AI, ZK, gaming, rendering, HPC, and scientific workloads.

Key features:
- Decentralized GPU nodes worldwide
- Staking and proof mechanisms for verification
- Low-cost, high-performance compute
- Support for AI inference, training, and more

AIDP enables builders to access GPU compute without centralized cloud providers.""",

    "Explain decentralized GPU compute": """Decentralized GPU compute distributes processing across a network of independent GPU providers instead of centralized data centers.

Benefits:
1. **Cost Efficiency**: 40-60% cheaper than AWS/GCP
2. **No Vendor Lock-in**: Use any provider in the network
3. **Redundancy**: No single point of failure
4. **Global Distribution**: Lower latency worldwide
5. **Verifiable**: Cryptographic proofs ensure computation integrity

AIDP implements this through staking, proofs, and decentralized routing.""",

    "default": """I'm Purple Squirrel R1, a fine-tuned model running on AIDP's decentralized GPU network. I can help with questions about AI, blockchain, and technology. What would you like to know?"""
}


def simulate_streaming(text: str, delay: float = 0.02):
    """Simulate streaming output"""
    words = text.split()
    for i, word in enumerate(words):
        print(word, end=" ", flush=True)
        if i % 5 == 4:  # Occasional pause
            time.sleep(delay * 2)
        else:
            time.sleep(delay)
    print()


def demo_chat_completion():
    """Demo: Single chat completion"""
    print("=" * 60)
    print("Demo 1: Chat Completion")
    print("=" * 60)
    print()

    prompt = "What is AIDP?"
    print(f"User: {prompt}")
    print()
    print("Assistant: ", end="")

    response = DEMO_RESPONSES.get(prompt, DEMO_RESPONSES["default"])
    simulate_streaming(response)

    print()
    print(f"[Tokens: ~{len(response.split())} | Latency: ~180ms]")
    print()


def demo_streaming():
    """Demo: Streaming response"""
    print("=" * 60)
    print("Demo 2: Streaming Response")
    print("=" * 60)
    print()

    prompt = "Explain decentralized GPU compute"
    print(f"User: {prompt}")
    print()
    print("Assistant (streaming): ", end="")

    response = DEMO_RESPONSES.get(prompt, DEMO_RESPONSES["default"])
    simulate_streaming(response, delay=0.03)

    print()


def demo_batch():
    """Demo: Batch inference"""
    print("=" * 60)
    print("Demo 3: Batch Inference")
    print("=" * 60)
    print()

    prompts = [
        "What is AIDP?",
        "Explain decentralized GPU compute",
        "What AI models are available?"
    ]

    print(f"Processing {len(prompts)} prompts in parallel...")
    print()

    start = time.time()

    for i, prompt in enumerate(prompts):
        response = DEMO_RESPONSES.get(prompt, DEMO_RESPONSES["default"])
        print(f"[{i+1}] Q: {prompt[:40]}...")
        print(f"    A: {response[:80]}...")
        print()

    elapsed = time.time() - start
    print(f"[Batch complete: {len(prompts)} prompts in {elapsed:.2f}s]")
    print()


def demo_openai_compat():
    """Demo: OpenAI compatibility"""
    print("=" * 60)
    print("Demo 4: OpenAI-Compatible API")
    print("=" * 60)
    print()

    print("```python")
    print("import openai")
    print()
    print("client = openai.OpenAI(")
    print('    base_url="https://neural-cloud.aidp.store/v1",')
    print('    api_key="your-aidp-api-key"')
    print(")")
    print()
    print("response = client.chat.completions.create(")
    print('    model="purple-squirrel-r1",')
    print("    messages=[")
    print('        {"role": "user", "content": "What is AIDP?"}')
    print("    ]")
    print(")")
    print()
    print("print(response.choices[0].message.content)")
    print("```")
    print()

    # Simulated response
    api_response = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "purple-squirrel-r1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": DEMO_RESPONSES["What is AIDP?"][:100] + "..."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 85,
            "total_tokens": 97
        }
    }

    print("Response:")
    print(json.dumps(api_response, indent=2))
    print()


def main():
    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║           AIDP Neural Cloud - Demo                        ║")
    print("║   Distributed LLM Inference on Decentralized GPUs         ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    print("Model: Purple Squirrel R1 (8B, 4-bit quantized)")
    print("Network: AIDP Decentralized GPU Network")
    print()

    demo_chat_completion()
    demo_streaming()
    demo_batch()
    demo_openai_compat()

    print("=" * 60)
    print("Demo complete!")
    print()
    print("Deploy your own models on AIDP: https://aidp.store")
    print("GitHub: https://github.com/PurpleSquirrelMedia/aidp-neural-cloud")
    print("=" * 60)


if __name__ == "__main__":
    main()
