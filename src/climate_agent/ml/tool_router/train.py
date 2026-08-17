import json

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from climate_agent.ml.tool_router.evaluate import evaluate, print_report
from climate_agent.ml.tool_router.generate_data import SYSTEM_PROMPT, TRAIN_FILE
from climate_agent.ml.tool_router.registry import ADAPTER_DIR, save_adapter

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 4


def _load_train_examples() -> list[dict]:
    return [json.loads(line) for line in TRAIN_FILE.read_text().splitlines() if line]


def _tokenize_example(tokenizer, example: dict) -> dict:
    """Format one (query, target) pair as a chat prompt + target completion, with loss masked
    to only the completion tokens (standard SFT masking).

    Args: tokenizer — the model's tokenizer. example — {"query": ..., "target": {...}}.
    Returns: {"input_ids", "attention_mask", "labels"}.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": example["query"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    completion = json.dumps(example["target"]) + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def build_dataset(tokenizer) -> Dataset:
    """Tokenized, loss-masked training dataset from TRAIN_FILE."""
    examples = _load_train_examples()
    tokenized = [_tokenize_example(tokenizer, ex) for ex in examples]
    return Dataset.from_list(tokenized)


def train() -> None:
    """Fine-tune BASE_MODEL_ID with LoRA on the router extraction task, tracked via MLflow."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype="auto")

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    train_dataset = build_dataset(tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=str(ADAPTER_DIR / "_checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    with mlflow.start_run(run_name="router_lora_qwen2.5-0.5b"):
        mlflow.log_params(
            {
                "base_model": BASE_MODEL_ID,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_target_modules": ",".join(LORA_TARGET_MODULES),
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "train_examples": len(train_dataset),
            }
        )

        print("Baseline eval (before fine-tuning)...")
        model.eval()
        baseline_results = evaluate(model, tokenizer)
        print_report("BASELINE", baseline_results)
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_results["field_accuracy"].items()})
        mlflow.log_metric("baseline_exact_match", baseline_results["exact_match"])

        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=collator)
        train_result = trainer.train()
        mlflow.log_metric("final_train_loss", train_result.training_loss)

        print("Fine-tuned eval...")
        model.eval()
        finetuned_results = evaluate(model, tokenizer)
        print_report("FINE-TUNED", finetuned_results)
        mlflow.log_metrics({f"finetuned_{k}": v for k, v in finetuned_results["field_accuracy"].items()})
        mlflow.log_metric("finetuned_exact_match", finetuned_results["exact_match"])

        save_adapter(model)
        mlflow.log_artifacts(str(ADAPTER_DIR), artifact_path="adapter")


if __name__ == "__main__":
    train()
