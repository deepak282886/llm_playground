import json
import random
import os
from config import SKILLS
from prompts_skills import SKILL_PROMPTS

INPUT_DIR = "train_data"

def _save_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_skill(skill_name):
    raw_file = os.path.join(INPUT_DIR, skill_name, "raw_valid.jsonl")

    config         = SKILL_PROMPTS[skill_name]
    system_prompt  = config["system_prompt"]

    examples = []
    with open(raw_file, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line.strip())
                if row.get("valid") and row.get("annotation"):
                    examples.append({
                        "messages": [
                            {"role": "system",    "content": system_prompt},
                            {"role": "user",      "content": row["user_message"]},
                            {"role": "assistant", "content": json.dumps(row["annotation"], ensure_ascii=False)},
                        ]
                    })
            except Exception:
                continue

    if not examples:
        print(f"  {skill_name}: no valid examples found.")
        return None

    random.shuffle(examples)
    split_idx  = int(len(examples) * 0.9)
    train_data = examples[:split_idx]
    val_data   = examples[split_idx:]

    out_dir = f"processed_data/{skill_name}"
    os.makedirs(out_dir, exist_ok=True)
    _save_jsonl(train_data, f"{out_dir}/train.jsonl")
    _save_jsonl(val_data,   f"{out_dir}/val.jsonl")

    print(f"  {skill_name}: train={len(train_data)} | val={len(val_data)}")
    return len(train_data), len(val_data)


if __name__ == "__main__":
    os.makedirs("processed_data", exist_ok=True)
    all_metadata = {}

    for skill_name in SKILLS:
        result = convert_skill(skill_name)
        if result:
            train_count, val_count = result
            all_metadata[skill_name] = {
                "status":      "success",
                "train_count": train_count,
                "val_count":   val_count,
                "train_file":  f"processed_data/{skill_name}/train.jsonl",
                "val_file":    f"processed_data/{skill_name}/val.jsonl",
            }
        else:
            all_metadata[skill_name] = {"status": "failed"}

    with open("processed_data/metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    successful = sum(1 for v in all_metadata.values() if v["status"] == "success")
    print(f"\nDone: {successful}/{len(SKILLS)} skills converted.")