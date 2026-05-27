# Skill-Specialized Small Language Models

The idea is simple: instead of using one large model for everything, train a set of small 0.5B models where each is an expert in one narrow conversational skill. A fine-tuned 0.5B specialist should match or beat a 7B+ generalist on its specific task at a fraction of the cost and memory.

## How It Works

A router model receives the user query and dispatches it to the correct specialist. Each specialist handles only what it was trained for and returns a structured JSON response.

```
User Query → Router → Skill Specialist → Structured Response
```

## Skills

| Skill | What It Does |
|---|---|
| intent_understanding | Classifies intent, urgency, sentiment, entities |
| context_tracking | Detects references, follow-ups, topic shifts |
| tone_matching | Reads tone and generates a matching response |
| clarity | Simplifies complex text for a target audience |
| dialogue | Generates natural follow-up questions |
| empathy | Responds to emotional messages with appropriate support |
| conciseness | Strips verbose text down to its core meaning |
| structure | Organises unstructured content into a clear format |
| writing | Fixes grammar, style, and writing quality |
| router | Decides which skill(s) to invoke for a given query |

## Pipeline

### 1. Data Generation (`generate_train.py`)
- Loads seed messages from CLINC150, Banking77, OpenAssistant, Alpaca, Dolly
- Buckets messages into neutral / urgent / emotional with skill-specific ratios
- Annotates each message using `openai/gpt-oss-20b` via Together AI
- Validates every annotation — required fields, enum checks, JSON structure
- Outputs raw valid annotations to `train_data/<skill>/raw_valid.jsonl`
- Target: 5,000 valid examples per skill

Config constants live in `config.py`. Skill prompts and validators live in `prompts_skills.py`.

### 2. Conversion (`convert_to_processed.py`)
- Reads `train_data/<skill>/raw_valid.jsonl`
- Converts to 3-turn chat format (system / user / assistant)
- Splits 90/10 train/val
- Writes to `processed_data/<skill>/train.jsonl` and `val.jsonl`
- Generates `processed_data/metadata.json` for the training script

### 3. Training (`train_all_skills.py`)
- Base model: `Qwen2.5-0.5B-Instruct`
- Fine-tuning: LoRA via Unsloth (`r=64, alpha=128`)
- One separate model trained per skill
- Saves LoRA adapters + merged 16bit model per skill
- Reads `processed_data/metadata.json` and only trains skills with successful data gen

### 4. Evaluation (`test_skills.py`)
- 5 handcrafted test cases per skill
- Runs each test through the fine-tuned specialist locally
- Runs the same test through `gpt-oss-20b` and `Qwen2.5-7B-Instruct-Turbo` via API
- LLM-as-judge scores all responses on accuracy, format, relevance (out of 30)
- Prints win/loss/tie table per skill and saves full results to `evaluation_results.json`

## Project Structure

```
├── config.py                # API key, model, constants
├── prompts_skills.py        # Per-skill prompts, validators, bucket weights
├── generate_train.py        # Data generation pipeline
├── convert_to_processed.py  # Convert raw annotations to training format
├── train_all_skills.py      # Fine-tuning pipeline
├── test_skills.py           # Evaluation script
├── train_data/
│   └── <skill>/
│       ├── raw_valid.jsonl
│       └── raw_invalid.jsonl
├── processed_data/
│   ├── metadata.json
│   └── <skill>/
│       ├── train.jsonl
│       └── val.jsonl
├── finetuned_models/
    └── qwen25-0.5b-<skill>-v2-merged/
```

