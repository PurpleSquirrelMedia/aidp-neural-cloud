"""
AIDP Neural Cloud - OpenAI-Compatible LLM Inference API
Distributed inference across AIDP decentralized GPU network
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from aidp_client import AIDPInferenceClient
from load_balancer import LoadBalancer
from models import AVAILABLE_MODELS, ModelConfig


# Request/Response Models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "purple-squirrel-r1"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    stream: bool = False
    stop: Optional[list[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class BatchRequest(BaseModel):
    prompts: list[str]
    model: str = "purple-squirrel-r1"
    max_tokens: int = 256
    temperature: float = 0.7


class BatchResponse(BaseModel):
    results: list[dict]
    count: int
    total_tokens: int
    processing_time_ms: int


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


# Global instances
load_balancer: Optional[LoadBalancer] = None
aidp_client: Optional[AIDPInferenceClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global load_balancer, aidp_client

    print("Initializing AIDP Neural Cloud...")

    # Initialize AIDP client
    aidp_client = AIDPInferenceClient()

    # Initialize load balancer with health checking
    load_balancer = LoadBalancer(aidp_client)
    await load_balancer.start()

    print(f"Neural Cloud ready - {len(AVAILABLE_MODELS)} models available")

    yield

    # Cleanup
    await load_balancer.stop()
    await aidp_client.close()
    print("Neural Cloud shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="AIDP Neural Cloud",
    description="Distributed LLM inference on AIDP decentralized GPU network",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    nodes = await load_balancer.get_healthy_nodes() if load_balancer else []
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "active_nodes": len(nodes),
        "models_available": list(AVAILABLE_MODELS.keys())
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    models = []
    for model_id, config in AVAILABLE_MODELS.items():
        models.append({
            "id": model_id,
            "object": "model",
            "created": config.created_at,
            "owned_by": config.owned_by,
            "permission": [],
            "root": model_id,
            "parent": None,
        })

    return {"object": "list", "data": models}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get model details"""
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    config = AVAILABLE_MODELS[model_id]
    return {
        "id": model_id,
        "object": "model",
        "created": config.created_at,
        "owned_by": config.owned_by,
        "permission": [],
        "root": model_id,
        "parent": None,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint

    Routes requests to AIDP GPU nodes for inference
    """
    # Validate model
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} not available. Use /v1/models to list available models."
        )

    model_config = AVAILABLE_MODELS[request.model]

    # Get best node for this request
    node = await load_balancer.get_best_node(
        model=request.model,
        estimated_tokens=request.max_tokens
    )

    if not node:
        raise HTTPException(
            status_code=503,
            detail="No GPU nodes available. Please try again later."
        )

    # Build prompt from messages
    prompt = _format_messages(request.messages, model_config)

    if request.stream:
        # Streaming response
        return StreamingResponse(
            _stream_response(node, prompt, request, model_config),
            media_type="text/event-stream"
        )

    # Non-streaming response
    start_time = time.time()

    result = await aidp_client.inference(
        node_id=node["id"],
        model=request.model,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Build response
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    return ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result["text"]),
                finish_reason=result.get("finish_reason", "stop")
            )
        ],
        usage=Usage(
            prompt_tokens=result.get("prompt_tokens", len(prompt.split())),
            completion_tokens=result.get("completion_tokens", len(result["text"].split())),
            total_tokens=result.get("total_tokens", len(prompt.split()) + len(result["text"].split()))
        )
    )


async def _stream_response(
    node: dict,
    prompt: str,
    request: ChatCompletionRequest,
    model_config: ModelConfig
) -> AsyncGenerator[str, None]:
    """Stream inference response"""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async for chunk in aidp_client.inference_stream(
        node_id=node["id"],
        model=request.model,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop
    ):
        data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk["text"]} if chunk.get("text") else {},
                "finish_reason": chunk.get("finish_reason")
            }]
        }
        yield f"data: {data}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/batch")
async def batch_inference(request: BatchRequest):
    """
    Batch inference endpoint for processing multiple prompts efficiently

    Distributes prompts across available AIDP GPU nodes
    """
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not available")

    if len(request.prompts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 prompts per batch")

    model_config = AVAILABLE_MODELS[request.model]
    start_time = time.time()

    # Get available nodes
    nodes = await load_balancer.get_healthy_nodes()
    if not nodes:
        raise HTTPException(status_code=503, detail="No GPU nodes available")

    # Distribute prompts across nodes
    results = []
    total_tokens = 0

    # Process in parallel batches
    tasks = []
    for i, prompt in enumerate(request.prompts):
        node = nodes[i % len(nodes)]  # Round-robin distribution
        task = aidp_client.inference(
            node_id=node["id"],
            model=request.model,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        tasks.append((prompt, task))

    # Gather results
    for prompt, task in tasks:
        try:
            result = await task
            results.append({
                "prompt": prompt,
                "response": result["text"],
                "tokens": result.get("total_tokens", 0)
            })
            total_tokens += result.get("total_tokens", 0)
        except Exception as e:
            results.append({
                "prompt": prompt,
                "error": str(e)
            })

    elapsed_ms = int((time.time() - start_time) * 1000)

    return BatchResponse(
        results=results,
        count=len(results),
        total_tokens=total_tokens,
        processing_time_ms=elapsed_ms
    )


@app.get("/metrics")
async def get_metrics():
    """Get inference metrics and node statistics"""
    nodes = await load_balancer.get_healthy_nodes() if load_balancer else []

    return {
        "active_nodes": len(nodes),
        "total_requests": load_balancer.total_requests if load_balancer else 0,
        "avg_latency_ms": load_balancer.avg_latency_ms if load_balancer else 0,
        "nodes": [
            {
                "id": n["id"],
                "gpu": n.get("gpu_type", "unknown"),
                "load": n.get("current_load", 0),
                "latency_ms": n.get("latency_ms", 0)
            }
            for n in nodes
        ]
    }


def _format_messages(messages: list[ChatMessage], model_config: ModelConfig) -> str:
    """Format chat messages into model-specific prompt format"""
    if model_config.chat_template:
        # Use model-specific template
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"{model_config.system_prefix}{msg.content}{model_config.system_suffix}"
            elif msg.role == "user":
                formatted += f"{model_config.user_prefix}{msg.content}{model_config.user_suffix}"
            elif msg.role == "assistant":
                formatted += f"{model_config.assistant_prefix}{msg.content}{model_config.assistant_suffix}"
        formatted += model_config.assistant_prefix
        return formatted

    # Default format
    prompt = ""
    for msg in messages:
        prompt += f"{msg.role}: {msg.content}\n"
    prompt += "assistant: "
    return prompt


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
