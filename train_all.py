import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import torch._dynamo
import gc
from datasets import load_dataset

torch._dynamo.config.suppress_errors = True

import pathlib
original_read_text = pathlib.Path.read_text
def safe_read_text(self, encoding="utf-8", errors="strict"):
    try:
        return original_read_text(self, encoding=encoding, errors=errors)
    except (UnicodeDecodeError, UnicodeError):
        return original_read_text(self, encoding="utf-8", errors="replace")
pathlib.Path.read_text = safe_read_text

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

MODEL_NAME   = "unsloth/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LEN  = 2048
DTYPE        = None
LOAD_4BIT    = True

LORA_R       = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

LR           = 2e-4
BATCH_SIZE   = 4
GRAD_ACCUM   = 4
EPOCHS       = 3
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
LOGGING_STEPS= 10
SAVE_STEPS   = 500
OUTPUT_DIR   = "finetuned_models_qwen"

SKILLS = [
    "intent_understanding",
    "context_tracking",
    "tone_matching",
    "clarity",
    "dialogue",
    "empathy",
    "conciseness",
    "structure",
    "writing",
    "router",
]


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_skill_data(skill_name):
    train_file = f"processed_data/{skill_name}/train.jsonl"
    val_file   = f"processed_data/{skill_name}/val.jsonl"
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Missing: {train_file}")
    train_ds = load_dataset("json", data_files=train_file, split="train")
    val_ds   = load_dataset("json", data_files=val_file,   split="train")
    return train_ds, val_ds


def format_messages(messages):
    parts = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"### System:\n{content}")
        elif role == "user":
            parts.append(f"### User:\n{content}")
        elif role == "assistant":
            parts.append(f"### Assistant:\n{content}")
    return "\n\n".join(parts)


def get_formatting_func(dataset):
    cols = dataset.column_names
    if "messages" in cols:
        def fmt(examples):
            if isinstance(examples["messages"][0], dict):
                batch = [examples["messages"]]
            else:
                batch = examples["messages"]
            return [format_messages(msgs) for msgs in batch]
        return fmt
    raise ValueError(f"Expected 'messages' column, got: {cols}")


def finetune_skill(skill_name, train_ds, val_ds):
    print(f"\n{'='*60}")
    print(f"SKILL: {skill_name.upper()}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"{'='*60}")

    cleanup()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = MODEL_NAME,
        max_seq_length= MAX_SEQ_LEN,
        dtype         = DTYPE,
        load_in_4bit  = LOAD_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                         = LORA_R,
        lora_alpha                = LORA_ALPHA,
        lora_dropout              = LORA_DROPOUT,
        target_modules            = TARGET_MODULES,
        bias                      = "none",
        use_gradient_checkpointing= "unsloth",
        random_state              = 42,
    )

    formatting_func = get_formatting_func(train_ds)

    lora_dir   = os.path.join(OUTPUT_DIR, f"qwen25-0.5b-{skill_name}-v2")
    merged_dir = os.path.join(OUTPUT_DIR, f"qwen25-0.5b-{skill_name}-v2-merged")

    training_args = SFTConfig(
        output_dir                  = lora_dir,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_steps                = WARMUP_STEPS,
        num_train_epochs            = EPOCHS,
        learning_rate               = LR,
        weight_decay                = WEIGHT_DECAY,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = LOGGING_STEPS,
        save_steps                  = SAVE_STEPS,
        eval_steps                  = SAVE_STEPS,
        eval_strategy               = "steps",
        save_strategy               = "steps",
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        optim                       = "adamw_8bit",
        gradient_checkpointing      = True,
        report_to                   = "none",
        max_seq_length              = MAX_SEQ_LEN,
        packing                     = False,
    )

    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        args            = training_args,
        formatting_func = formatting_func,
    )

    trainer.train()

    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    print(f"Saving merged 16bit model...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print(f"Done: {skill_name}")

    del model, tokenizer, trainer
    cleanup()

    return lora_dir, merged_dir


if __name__ == "__main__":
    metadata_file = "processed_data/metadata.json"
    if not os.path.exists(metadata_file):
        print("metadata.json not found. Run data generation first.")
        exit(1)

    with open(metadata_file, encoding="utf-8") as f:
        metadata = json.load(f)

    available_skills = [s for s in SKILLS if metadata.get(s, {}).get("status") == "success"]
    print(f"Skills to train: {available_skills}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    for i, skill_name in enumerate(available_skills, 1):
        print(f"\n[{i}/{len(available_skills)}] {skill_name}")
        try:
            train_ds, val_ds       = load_skill_data(skill_name)
            lora_dir, merged_dir   = finetune_skill(skill_name, train_ds, val_ds)
            results[skill_name]    = {
                "status":       "success",
                "lora_model":   lora_dir,
                "merged_model": merged_dir,
                "train_count":  len(train_ds),
                "val_count":    len(val_ds),
            }
        except Exception as e:
            print(f"FAILED {skill_name}: {e}")
            import traceback; traceback.print_exc()
            results[skill_name] = {"status": "failed", "error": str(e)}
            cleanup()
            continue

    results_file = os.path.join(OUTPUT_DIR, "training_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    successful = sum(1 for v in results.values() if v["status"] == "success")
    print(f"Successful : {successful}/{len(available_skills)}")
    for skill, meta in results.items():
        status = "✓" if meta["status"] == "success" else "✗"
        print(f"  {status} {skill:<25} {meta.get('error', '')}")