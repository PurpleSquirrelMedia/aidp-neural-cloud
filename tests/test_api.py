"""
Tests for AIDP Neural Cloud API
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import AVAILABLE_MODELS, get_model_config, list_model_ids, get_models_for_vram


class TestModels:
    """Test model configurations"""

    def test_all_models_exist(self):
        """Test all expected models are defined"""
        expected = [
            "purple-squirrel-r1",
            "llama-3.1-8b",
            "mistral-7b",
            "qwen2-7b",
            "deepseek-coder-6.7b",
            "phi-3-mini"
        ]
        for model_id in expected:
            assert model_id in AVAILABLE_MODELS, f"Missing model: {model_id}"

    def test_model_has_required_fields(self):
        """Test models have required fields"""
        required_fields = [
            "id", "name", "description", "owned_by", "created_at",
            "parameters", "quantization", "context_length", "min_vram_gb"
        ]
        for model_id, config in AVAILABLE_MODELS.items():
            for field in required_fields:
                assert hasattr(config, field), f"Model {model_id} missing field: {field}"

    def test_get_model_config(self):
        """Test get_model_config function"""
        config = get_model_config("purple-squirrel-r1")
        assert config is not None
        assert config.id == "purple-squirrel-r1"
        assert config.parameters == "8B"

        # Test nonexistent model
        config = get_model_config("nonexistent")
        assert config is None

    def test_list_model_ids(self):
        """Test list_model_ids function"""
        ids = list_model_ids()
        assert len(ids) >= 6
        assert "purple-squirrel-r1" in ids

    def test_get_models_for_vram(self):
        """Test get_models_for_vram function"""
        # 3GB VRAM should only get phi-3-mini
        models_3gb = get_models_for_vram(3)
        assert "phi-3-mini" in models_3gb
        assert "llama-3.1-8b" not in models_3gb

        # 6GB VRAM should get most models
        models_6gb = get_models_for_vram(6)
        assert "purple-squirrel-r1" in models_6gb
        assert "mistral-7b" in models_6gb

    def test_purple_squirrel_config(self):
        """Test Purple Squirrel R1 configuration"""
        config = get_model_config("purple-squirrel-r1")
        assert config.owned_by == "purplesquirrelnetworks"
        assert config.quantization == "4bit"
        assert config.min_vram_gb == 6
        assert config.chat_template is True

    def test_chat_templates(self):
        """Test chat template formatting"""
        for model_id, config in AVAILABLE_MODELS.items():
            if config.chat_template:
                # Should have prefix/suffix defined
                assert config.user_prefix is not None
                assert config.assistant_prefix is not None


class TestLoadBalancer:
    """Test load balancer functionality"""

    def test_node_stats_dataclass(self):
        """Test NodeStats dataclass"""
        from load_balancer import NodeStats

        stats = NodeStats(
            id="test-node",
            gpu_type="A10G",
            vram_gb=24
        )

        assert stats.id == "test-node"
        assert stats.gpu_type == "A10G"
        assert stats.healthy is True  # Default
        assert stats.current_load == 0.0  # Default


class TestAIDPClient:
    """Test AIDP client functionality"""

    def test_client_initialization(self):
        """Test client can be initialized"""
        from aidp_client import AIDPInferenceClient

        client = AIDPInferenceClient(
            api_url="https://test.api.aidp.store",
            api_key="test-key",
            wallet="test-wallet"
        )

        assert client.api_url == "https://test.api.aidp.store"
        assert client.api_key == "test-key"
        assert client.wallet == "test-wallet"

    def test_inference_error_class(self):
        """Test AIDPInferenceError"""
        from aidp_client import AIDPInferenceError

        error = AIDPInferenceError("Test error")
        assert str(error) == "Test error"


class TestAPIEndpoints:
    """Test API endpoint definitions"""

    def test_chat_message_model(self):
        """Test ChatMessage model"""
        from api import ChatMessage

        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_completion_request(self):
        """Test ChatCompletionRequest model"""
        from api import ChatCompletionRequest, ChatMessage

        request = ChatCompletionRequest(
            model="purple-squirrel-r1",
            messages=[ChatMessage(role="user", content="Hello")]
        )

        assert request.model == "purple-squirrel-r1"
        assert len(request.messages) == 1
        assert request.max_tokens == 256  # Default
        assert request.temperature == 0.7  # Default
        assert request.stream is False  # Default

    def test_chat_completion_request_validation(self):
        """Test request validation"""
        from api import ChatCompletionRequest, ChatMessage

        # Max tokens limit
        request = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=4096
        )
        assert request.max_tokens == 4096

        # Temperature bounds
        request = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.0
        )
        assert request.temperature == 0.0

    def test_batch_request(self):
        """Test BatchRequest model"""
        from api import BatchRequest

        request = BatchRequest(
            prompts=["Hello", "World"],
            model="purple-squirrel-r1"
        )

        assert len(request.prompts) == 2
        assert request.model == "purple-squirrel-r1"


class TestResponseModels:
    """Test response model definitions"""

    def test_chat_completion_response(self):
        """Test ChatCompletionResponse model"""
        from api import ChatCompletionResponse, ChatCompletionChoice, ChatMessage, Usage

        response = ChatCompletionResponse(
            id="test-123",
            created=1234567890,
            model="purple-squirrel-r1",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hello!"),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=5,
                completion_tokens=2,
                total_tokens=7
            )
        )

        assert response.id == "test-123"
        assert response.object == "chat.completion"
        assert len(response.choices) == 1
        assert response.usage.total_tokens == 7

    def test_batch_response(self):
        """Test BatchResponse model"""
        from api import BatchResponse

        response = BatchResponse(
            results=[{"prompt": "Hello", "response": "Hi!"}],
            count=1,
            total_tokens=10,
            processing_time_ms=100
        )

        assert response.count == 1
        assert response.total_tokens == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
