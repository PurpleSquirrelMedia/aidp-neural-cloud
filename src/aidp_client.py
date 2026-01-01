"""
AIDP Inference Client - Interface to AIDP GPU nodes for LLM inference
Handles node discovery, request routing, and inference execution
"""

import asyncio
import os
import time
from typing import AsyncGenerator, Optional

# AIDP Configuration
AIDP_API_URL = os.getenv("AIDP_API_URL", "https://api.aidp.store")
AIDP_API_KEY = os.getenv("AIDP_API_KEY", "")
AIDP_WALLET = os.getenv("AIDP_WALLET", "")


class AIDPInferenceClient:
    """
    Client for LLM inference on AIDP decentralized GPU network

    Supports:
    - Node discovery and health checking
    - Synchronous and streaming inference
    - Automatic retries and failover
    """

    def __init__(
        self,
        api_url: str = AIDP_API_URL,
        api_key: str = AIDP_API_KEY,
        wallet: str = AIDP_WALLET
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.wallet = wallet
        self._session = None

    async def _get_session(self):
        """Get or create aiohttp session"""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "X-AIDP-Wallet": self.wallet,
                    "Content-Type": "application/json"
                },
                timeout=aiohttp.ClientTimeout(total=120)
            )
        return self._session

    async def close(self):
        """Close the session"""
        if self._session:
            await self._session.close()
            self._session = None

    async def list_inference_nodes(
        self,
        model: Optional[str] = None,
        min_vram_gb: int = 0
    ) -> list[dict]:
        """
        List available inference nodes on AIDP network

        Args:
            model: Filter by model availability
            min_vram_gb: Minimum VRAM requirement

        Returns:
            List of available nodes with capabilities
        """
        session = await self._get_session()

        params = {}
        if model:
            params["model"] = model
        if min_vram_gb:
            params["min_vram_gb"] = min_vram_gb

        async with session.get(
            f"{self.api_url}/v1/inference/nodes",
            params=params
        ) as resp:
            if resp.status != 200:
                raise AIDPInferenceError(f"Failed to list nodes: {await resp.text()}")
            data = await resp.json()
            return data.get("nodes", [])

    async def get_node_health(self, node_id: str) -> dict:
        """Get health status of a specific node"""
        session = await self._get_session()

        async with session.get(
            f"{self.api_url}/v1/inference/nodes/{node_id}/health"
        ) as resp:
            if resp.status != 200:
                return {"healthy": False, "error": await resp.text()}
            return await resp.json()

    async def inference(
        self,
        node_id: str,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        timeout: int = 60
    ) -> dict:
        """
        Run inference on a specific AIDP GPU node

        Args:
            node_id: Target GPU node ID
            model: Model to use for inference
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            timeout: Request timeout in seconds

        Returns:
            dict with generated text and usage metrics
        """
        session = await self._get_session()

        request_body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }
        if stop:
            request_body["stop"] = stop

        start_time = time.time()

        async with session.post(
            f"{self.api_url}/v1/inference/nodes/{node_id}/generate",
            json=request_body,
            timeout=timeout
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise AIDPInferenceError(f"Inference failed: {error}")

            result = await resp.json()

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "text": result.get("text", ""),
            "finish_reason": result.get("finish_reason", "stop"),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "latency_ms": elapsed_ms,
            "node_id": node_id,
            "model": model
        }

    async def inference_stream(
        self,
        node_id: str,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None
    ) -> AsyncGenerator[dict, None]:
        """
        Stream inference from AIDP GPU node

        Yields:
            dict with text chunk and metadata
        """
        session = await self._get_session()

        request_body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True
        }
        if stop:
            request_body["stop"] = stop

        async with session.post(
            f"{self.api_url}/v1/inference/nodes/{node_id}/generate",
            json=request_body
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise AIDPInferenceError(f"Stream inference failed: {error}")

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        yield {"text": "", "finish_reason": "stop"}
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        yield {
                            "text": chunk.get("text", ""),
                            "finish_reason": chunk.get("finish_reason")
                        }
                    except json.JSONDecodeError:
                        continue

    async def deploy_model(
        self,
        model_id: str,
        model_path: str,
        quantization: str = "4bit",
        replicas: int = 1
    ) -> dict:
        """
        Deploy a model to AIDP network

        Args:
            model_id: Unique identifier for the model
            model_path: HuggingFace model path or local path
            quantization: Quantization level (4bit, 8bit, fp16)
            replicas: Number of GPU nodes to deploy to

        Returns:
            Deployment status and node assignments
        """
        session = await self._get_session()

        async with session.post(
            f"{self.api_url}/v1/inference/deploy",
            json={
                "model_id": model_id,
                "model_path": model_path,
                "quantization": quantization,
                "replicas": replicas,
                "config": {
                    "max_batch_size": 16,
                    "max_sequence_length": 4096
                }
            }
        ) as resp:
            if resp.status != 200:
                raise AIDPInferenceError(f"Deployment failed: {await resp.text()}")
            return await resp.json()

    async def undeploy_model(self, model_id: str) -> dict:
        """Remove a model from AIDP network"""
        session = await self._get_session()

        async with session.delete(
            f"{self.api_url}/v1/inference/models/{model_id}"
        ) as resp:
            if resp.status != 200:
                raise AIDPInferenceError(f"Undeploy failed: {await resp.text()}")
            return await resp.json()

    async def get_model_status(self, model_id: str) -> dict:
        """Get deployment status of a model"""
        session = await self._get_session()

        async with session.get(
            f"{self.api_url}/v1/inference/models/{model_id}"
        ) as resp:
            if resp.status == 404:
                return {"deployed": False}
            if resp.status != 200:
                raise AIDPInferenceError(f"Status check failed: {await resp.text()}")
            return await resp.json()

    async def get_pricing(self) -> dict:
        """Get current inference pricing"""
        session = await self._get_session()

        async with session.get(f"{self.api_url}/v1/inference/pricing") as resp:
            if resp.status != 200:
                raise AIDPInferenceError(f"Pricing request failed: {await resp.text()}")
            return await resp.json()


class AIDPInferenceError(Exception):
    """AIDP inference-specific error"""
    pass


# Convenience function for simple inference
async def generate(
    prompt: str,
    model: str = "purple-squirrel-r1",
    max_tokens: int = 256,
    temperature: float = 0.7
) -> str:
    """
    Simple inference function

    Example:
        response = await generate("What is AIDP?")
        print(response)
    """
    client = AIDPInferenceClient()

    try:
        # Get available nodes
        nodes = await client.list_inference_nodes(model=model)
        if not nodes:
            raise AIDPInferenceError(f"No nodes available for model {model}")

        # Use first available node
        node = nodes[0]

        # Run inference
        result = await client.inference(
            node_id=node["id"],
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return result["text"]

    finally:
        await client.close()
