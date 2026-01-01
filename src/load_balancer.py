"""
Load Balancer - Intelligent request routing across AIDP GPU nodes
Implements health checking, latency-based routing, and failover
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from aidp_client import AIDPInferenceClient


@dataclass
class NodeStats:
    """Statistics for a single node"""
    id: str
    gpu_type: str = "unknown"
    vram_gb: int = 0
    healthy: bool = True
    current_load: float = 0.0
    latency_ms: float = 0.0
    request_count: int = 0
    error_count: int = 0
    last_check: float = 0.0
    models: list = field(default_factory=list)


class LoadBalancer:
    """
    Intelligent load balancer for AIDP GPU inference nodes

    Features:
    - Health checking with automatic unhealthy node removal
    - Latency-based routing (prefer fastest nodes)
    - Load-aware distribution
    - Automatic failover on node failure
    """

    def __init__(
        self,
        client: AIDPInferenceClient,
        health_check_interval: int = 30,
        unhealthy_threshold: int = 3
    ):
        self.client = client
        self.health_check_interval = health_check_interval
        self.unhealthy_threshold = unhealthy_threshold

        self._nodes: dict[str, NodeStats] = {}
        self._health_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self.total_requests = 0
        self.total_latency_ms = 0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency across all requests"""
        if self.total_requests == 0:
            return 0
        return self.total_latency_ms / self.total_requests

    async def start(self):
        """Start the load balancer and health checking"""
        self._running = True
        await self._discover_nodes()
        self._health_task = asyncio.create_task(self._health_check_loop())
        print(f"Load balancer started with {len(self._nodes)} nodes")

    async def stop(self):
        """Stop the load balancer"""
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        print("Load balancer stopped")

    async def _discover_nodes(self):
        """Discover available GPU nodes"""
        try:
            nodes = await self.client.list_inference_nodes()
            for node in nodes:
                node_id = node["id"]
                self._nodes[node_id] = NodeStats(
                    id=node_id,
                    gpu_type=node.get("gpu_type", "unknown"),
                    vram_gb=node.get("vram_gb", 0),
                    healthy=True,
                    models=node.get("models", []),
                    last_check=time.time()
                )
        except Exception as e:
            print(f"Node discovery failed: {e}")

    async def _health_check_loop(self):
        """Periodic health check loop"""
        while self._running:
            await asyncio.sleep(self.health_check_interval)
            await self._check_all_nodes()

    async def _check_all_nodes(self):
        """Check health of all nodes"""
        for node_id in list(self._nodes.keys()):
            await self._check_node(node_id)

    async def _check_node(self, node_id: str):
        """Check health of a single node"""
        try:
            start = time.time()
            health = await self.client.get_node_health(node_id)
            latency = (time.time() - start) * 1000

            node = self._nodes.get(node_id)
            if node:
                node.healthy = health.get("healthy", False)
                node.latency_ms = latency
                node.current_load = health.get("load", 0)
                node.last_check = time.time()
                node.error_count = 0

        except Exception as e:
            node = self._nodes.get(node_id)
            if node:
                node.error_count += 1
                if node.error_count >= self.unhealthy_threshold:
                    node.healthy = False
                    print(f"Node {node_id} marked unhealthy after {node.error_count} errors")

    async def get_healthy_nodes(self) -> list[dict]:
        """Get list of healthy nodes"""
        healthy = []
        for node in self._nodes.values():
            if node.healthy:
                healthy.append({
                    "id": node.id,
                    "gpu_type": node.gpu_type,
                    "vram_gb": node.vram_gb,
                    "current_load": node.current_load,
                    "latency_ms": node.latency_ms,
                    "models": node.models
                })
        return healthy

    async def get_best_node(
        self,
        model: str,
        estimated_tokens: int = 256
    ) -> Optional[dict]:
        """
        Get the best node for a request using smart routing

        Routing strategy:
        1. Filter by model availability
        2. Filter by health status
        3. Score by: latency (40%) + load (40%) + recent success (20%)
        4. Return highest scoring node
        """
        candidates = []

        for node in self._nodes.values():
            # Skip unhealthy nodes
            if not node.healthy:
                continue

            # Check if node has the model (or no model filter)
            if model and node.models and model not in node.models:
                continue

            # Calculate score (lower is better)
            latency_score = node.latency_ms / 1000  # Normalize to seconds
            load_score = node.current_load
            error_score = node.error_count / 10  # Penalize error-prone nodes

            # Weighted score
            score = (
                latency_score * 0.4 +
                load_score * 0.4 +
                error_score * 0.2
            )

            candidates.append((score, node))

        if not candidates:
            # Try to discover new nodes
            await self._discover_nodes()
            return await self.get_best_node(model, estimated_tokens) if self._nodes else None

        # Sort by score (ascending) and return best
        candidates.sort(key=lambda x: x[0])
        best_node = candidates[0][1]

        return {
            "id": best_node.id,
            "gpu_type": best_node.gpu_type,
            "vram_gb": best_node.vram_gb,
            "latency_ms": best_node.latency_ms
        }

    def record_request(self, node_id: str, latency_ms: float, success: bool):
        """Record request metrics for a node"""
        self.total_requests += 1
        self.total_latency_ms += latency_ms

        node = self._nodes.get(node_id)
        if node:
            node.request_count += 1
            # Exponential moving average for latency
            alpha = 0.3
            node.latency_ms = alpha * latency_ms + (1 - alpha) * node.latency_ms

            if not success:
                node.error_count += 1

    async def failover(self, failed_node_id: str, model: str) -> Optional[dict]:
        """
        Get alternative node after failure

        Excludes the failed node from consideration
        """
        # Mark node as unhealthy temporarily
        if failed_node_id in self._nodes:
            self._nodes[failed_node_id].healthy = False

        # Get next best node
        return await self.get_best_node(model)

    def get_node_metrics(self) -> dict:
        """Get detailed metrics for all nodes"""
        return {
            "total_nodes": len(self._nodes),
            "healthy_nodes": sum(1 for n in self._nodes.values() if n.healthy),
            "total_requests": self.total_requests,
            "avg_latency_ms": self.avg_latency_ms,
            "nodes": {
                node_id: {
                    "healthy": node.healthy,
                    "gpu_type": node.gpu_type,
                    "vram_gb": node.vram_gb,
                    "latency_ms": node.latency_ms,
                    "load": node.current_load,
                    "requests": node.request_count,
                    "errors": node.error_count
                }
                for node_id, node in self._nodes.items()
            }
        }
