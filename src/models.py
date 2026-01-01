"""
Model Configurations for AIDP Neural Cloud
Defines available models and their inference settings
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a deployed model"""
    id: str
    name: str
    description: str
    owned_by: str
    created_at: int

    # Model specs
    parameters: str  # e.g., "8B", "70B"
    quantization: str  # e.g., "4bit", "8bit", "fp16"
    context_length: int
    min_vram_gb: int

    # Chat template
    chat_template: bool = True
    system_prefix: str = ""
    system_suffix: str = "\n"
    user_prefix: str = "User: "
    user_suffix: str = "\n"
    assistant_prefix: str = "Assistant: "
    assistant_suffix: str = "\n"

    # Generation defaults
    default_max_tokens: int = 256
    default_temperature: float = 0.7
    default_top_p: float = 1.0


# Available models on AIDP Neural Cloud
AVAILABLE_MODELS: dict[str, ModelConfig] = {
    "purple-squirrel-r1": ModelConfig(
        id="purple-squirrel-r1",
        name="Purple Squirrel R1",
        description="Fine-tuned DeepSeek R1 for Purple Squirrel Media domain knowledge",
        owned_by="purplesquirrelnetworks",
        created_at=1703376000,
        parameters="8B",
        quantization="4bit",
        context_length=4096,
        min_vram_gb=6,
        chat_template=True,
        system_prefix="<|System|>",
        system_suffix="<|end|>\n",
        user_prefix="<|User|>",
        user_suffix="<|end|>\n",
        assistant_prefix="<|Assistant|>",
        assistant_suffix="<|end|>\n",
        default_max_tokens=256,
        default_temperature=0.3,
    ),

    "llama-3.1-8b": ModelConfig(
        id="llama-3.1-8b",
        name="Llama 3.1 8B Instruct",
        description="Meta's Llama 3.1 8B instruction-tuned model",
        owned_by="meta-llama",
        created_at=1721692800,
        parameters="8B",
        quantization="4bit",
        context_length=8192,
        min_vram_gb=6,
        chat_template=True,
        system_prefix="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        system_suffix="<|eot_id|>",
        user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
        user_suffix="<|eot_id|>",
        assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        assistant_suffix="<|eot_id|>",
        default_max_tokens=512,
        default_temperature=0.6,
    ),

    "mistral-7b": ModelConfig(
        id="mistral-7b",
        name="Mistral 7B Instruct",
        description="Mistral AI's efficient 7B instruction-tuned model",
        owned_by="mistralai",
        created_at=1695859200,
        parameters="7B",
        quantization="4bit",
        context_length=8192,
        min_vram_gb=5,
        chat_template=True,
        system_prefix="",
        system_suffix="",
        user_prefix="[INST] ",
        user_suffix=" [/INST]",
        assistant_prefix="",
        assistant_suffix="</s>",
        default_max_tokens=512,
        default_temperature=0.7,
    ),

    "qwen2-7b": ModelConfig(
        id="qwen2-7b",
        name="Qwen2 7B Instruct",
        description="Alibaba's Qwen2 7B instruction-tuned model",
        owned_by="qwen",
        created_at=1717200000,
        parameters="7B",
        quantization="4bit",
        context_length=32768,
        min_vram_gb=5,
        chat_template=True,
        system_prefix="<|im_start|>system\n",
        system_suffix="<|im_end|>\n",
        user_prefix="<|im_start|>user\n",
        user_suffix="<|im_end|>\n",
        assistant_prefix="<|im_start|>assistant\n",
        assistant_suffix="<|im_end|>\n",
        default_max_tokens=512,
        default_temperature=0.7,
    ),

    "deepseek-coder-6.7b": ModelConfig(
        id="deepseek-coder-6.7b",
        name="DeepSeek Coder 6.7B",
        description="DeepSeek's code-specialized 6.7B model",
        owned_by="deepseek-ai",
        created_at=1698796800,
        parameters="6.7B",
        quantization="4bit",
        context_length=16384,
        min_vram_gb=5,
        chat_template=True,
        system_prefix="",
        system_suffix="\n",
        user_prefix="### Instruction:\n",
        user_suffix="\n",
        assistant_prefix="### Response:\n",
        assistant_suffix="\n",
        default_max_tokens=1024,
        default_temperature=0.0,  # Lower temp for code
    ),

    "phi-3-mini": ModelConfig(
        id="phi-3-mini",
        name="Phi-3 Mini 3.8B",
        description="Microsoft's efficient Phi-3 Mini model",
        owned_by="microsoft",
        created_at=1713744000,
        parameters="3.8B",
        quantization="4bit",
        context_length=4096,
        min_vram_gb=3,
        chat_template=True,
        system_prefix="<|system|>\n",
        system_suffix="<|end|>\n",
        user_prefix="<|user|>\n",
        user_suffix="<|end|>\n",
        assistant_prefix="<|assistant|>\n",
        assistant_suffix="<|end|>\n",
        default_max_tokens=256,
        default_temperature=0.7,
    ),
}


def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """Get configuration for a model by ID"""
    return AVAILABLE_MODELS.get(model_id)


def list_model_ids() -> list[str]:
    """List all available model IDs"""
    return list(AVAILABLE_MODELS.keys())


def get_models_for_vram(vram_gb: int) -> list[str]:
    """Get models that can run with given VRAM"""
    return [
        model_id for model_id, config in AVAILABLE_MODELS.items()
        if config.min_vram_gb <= vram_gb
    ]
