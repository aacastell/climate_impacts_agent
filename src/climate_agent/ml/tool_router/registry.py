from pathlib import Path

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = Path(__file__).parent / "adapter"


def save_adapter(model) -> None:
    """Save a PEFT model's trained LoRA adapter weights to ADAPTER_DIR.

    Args: model — a peft-wrapped model (result of get_peft_model + training).
    """
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))


def load_router_model() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the base router model with the trained LoRA adapter applied, for real inference.

    Returns: (model, tokenizer), model in eval mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype="auto")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    return model, tokenizer
