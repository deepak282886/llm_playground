import json
import requests
import random
import os
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import API_KEY, API_URL, OUTPUT_DIR, MODEL, MAX_TOKENS, TARGET_EXAMPLES, PARALLEL_WORKERS, SKILLS
from prompts_skills import SKILL_PROMPTS

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_seed_messages(target=15000):
    neutral_msgs   = []
    emotional_msgs = []
    urgent_msgs    = []

    try:
        ds   = load_dataset("clinc_oos", "plus", split="train")
        msgs = [{"text": item["text"], "source": "clinc150", "bucket": "neutral"}
                for item in ds if item.get("text")]
        neutral_msgs.extend(msgs)
    except Exception as e:
        print(e)

    try:
        for split in ["train", "test"]:
            ds   = load_dataset("banking77", split=split)
            msgs = [{"text": item["text"], "source": "banking77", "bucket": "urgent"}
                    for item in ds if item.get("text")]
            urgent_msgs.extend(msgs)
    except Exception as e:
        print(e)

    try:
        for split in ["train", "validation"]:
            ds = load_dataset("OpenAssistant/oasst1", split=split, streaming=True)
            for item in ds:
                if item.get("text") and item.get("role") == "prompter":
                    emotional_msgs.append({"text": item["text"], "source": "openassistant", "bucket": "emotional"})
    except Exception as e:
        print(e)

    try:
        ds   = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        msgs = []
        for i, item in enumerate(ds):
            if i >= 8000:
                break
            if item.get("instruction"):
                msgs.append({"text": item["instruction"], "source": "alpaca", "bucket": "neutral"})
        neutral_msgs.extend(msgs)
    except Exception as e:
        print(e)

    try:
        ds   = load_dataset("databricks/databricks-dolly-15k", split="train")
        msgs = [{"text": item["instruction"], "source": "dolly", "bucket": "neutral"}
                for item in ds if item.get("instruction")]
        neutral_msgs.extend(msgs)
    except Exception as e:
        print(e)


    random.shuffle(neutral_msgs)
    random.shuffle(urgent_msgs)
    random.shuffle(emotional_msgs)

    balanced = (
        neutral_msgs  [:int(target * 0.50)] +
        urgent_msgs   [:int(target * 0.30)] +
        emotional_msgs[:int(target * 0.20)]
    )

    random.shuffle(balanced)
    seen, unique = set(), []
    for m in balanced:
        t = m["text"].strip().lower()
        if t not in seen and len(t) > 5:
            seen.add(t)
            unique.append(m)

    print(f"Total: {len(unique)}")
    return unique[:target]


def annotate_message(message_data, skill_name, config):
    user_message = message_data["text"]
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       MODEL,
        "messages": [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user",   "content": config["user_prompt"](user_message)},
        ],
        "max_tokens":  MAX_TOKENS,
        "temperature": 0.2,
    }

    content = ""
    try:
    # if True:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            return None

        content = response.json()["choices"][0]["message"]["content"].strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)

        for field in config["required_fields"]:
            assert field in parsed, f"missing field: {field}"
        for field, valid_values in config["validators"].items():
            assert parsed[field] in valid_values, f"invalid {field}: {parsed[field]}"

        return {
            "user_message": user_message,
            "source":       message_data["source"],
            "bucket":       message_data["bucket"],
            "annotation":   parsed,
            "raw_output":   content,
            "valid":        True,
        }

    except Exception:
        return None


def annotate_all(messages, skill_name, config):
    valid_results   = []
    invalid_results = []

    skill_out    = os.path.join(OUTPUT_DIR, skill_name)
    os.makedirs(skill_out, exist_ok=True)
    output_file  = os.path.join(skill_out, "raw_valid.jsonl")
    invalid_file = os.path.join(skill_out, "raw_invalid.jsonl")

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(annotate_message, msg, skill_name, config): msg for msg in messages}

        with tqdm(total=len(futures), desc=skill_name, ncols=80) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    pbar.update(1)
                    continue

                if result["valid"]:
                    valid_results.append(result)
                else:
                    invalid_results.append(result)

                if len(valid_results) % 200 == 0 and valid_results:
                    _save_jsonl(valid_results,   output_file)
                    _save_jsonl(invalid_results, invalid_file)

                pbar.update(1)
                pbar.set_postfix({"valid": len(valid_results), "invalid": len(invalid_results)})

    _save_jsonl(valid_results,   output_file)
    _save_jsonl(invalid_results, invalid_file)
    return valid_results, invalid_results


def print_stats(valid_results, invalid_results, skill_name):
    total = len(valid_results) + len(invalid_results)
    rate  = len(valid_results) / total * 100 if total else 0
    print(f"\n{skill_name} — Valid: {len(valid_results)} ({rate:.1f}%) | Invalid: {len(invalid_results)}")
    for bucket in ["neutral", "urgent", "emotional"]:
        count = sum(1 for r in valid_results if r.get("bucket") == bucket)
        pct   = count / len(valid_results) * 100 if valid_results else 0
        print(f"  {bucket:<12} {count:>5} ({pct:4.1f}%)")


def _save_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    messages = load_seed_messages(target=TARGET_EXAMPLES + 2000)

    if len(messages) < 1000:
        print("Not enough seed messages.")
        exit(1)

    neutral_pool   = [m for m in messages if m["bucket"] == "neutral"]
    urgent_pool    = [m for m in messages if m["bucket"] == "urgent"]
    emotional_pool = [m for m in messages if m["bucket"] == "emotional"]

    all_metadata = {}

    for skill_name in SKILLS:
        print(skill_name)

        config  = SKILL_PROMPTS[skill_name]
        weights = config["bucket_weights"]
        n       = TARGET_EXAMPLES + 500

        sampled = (
            random.sample(neutral_pool,   min(int(n * weights["neutral"]),   len(neutral_pool)))   +
            random.sample(urgent_pool,    min(int(n * weights["urgent"]),    len(urgent_pool)))    +
            random.sample(emotional_pool, min(int(n * weights["emotional"]), len(emotional_pool)))
        )
        random.shuffle(sampled)

        try:
        # if True:
            valid_results, invalid_results = annotate_all(sampled, skill_name, config)
            print_stats(valid_results, invalid_results, skill_name)
            all_metadata[skill_name] = {"status": "success", "valid_count": len(valid_results)}

        except Exception as e:
            print("outer loop", e)
            all_metadata[skill_name] = {"status": "failed", "error": str(e)}
            continue