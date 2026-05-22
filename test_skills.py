import json
import requests
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import API_KEY, API_URL
from prompts_skills import SKILL_PROMPTS


BIG_MODELS = {
    "gpt-oss-20b":      "openai/gpt-oss-20b",
    "qwen2.5-7b-turbo": "Qwen/Qwen2.5-7B-Instruct-Turbo",
}

JUDGE_MODEL = "openai/gpt-oss-20b"

FINETUNED_DIR = "finetuned_models"

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

TEST_CASES = {
    "intent_understanding": [
        "I need to cancel my subscription right now.",
        "What are the best Python libraries for data visualization?",
        "My account has been charged twice this month!",
        "Hi there, how are you doing today?",
        "Can you compare React and Vue for a beginner project?",
    ],
    "context_tracking": [
        "What about the second option?",
        "Can you explain that again?",
        "How does it compare to what you said before?",
        "And what about the price?",
        "Does it work the same way?",
    ],
    "tone_matching": [
        "HEY!! MY ORDER STILL HASN'T ARRIVED AND IT'S BEEN 2 WEEKS!!!",
        "Good afternoon. I would like to inquire about your refund policy.",
        "omg this app is so confusing lol can u help??",
        "I am writing to formally request an update on my pending application.",
        "dude this is broken again wtf",
    ],
    "clarity": [
        "The polymorphic dispatch mechanism resolves method calls at runtime via vtable lookups.",
        "Myocardial infarction occurs due to prolonged ischemia causing irreversible cardiomyocyte necrosis.",
        "The indemnification clause stipulates that the licensee shall hold harmless the licensor.",
        "Quantitative easing involves central bank asset purchases to expand monetary base.",
        "The asynchronous paradigm allows non-blocking I/O operations via event loop delegation.",
    ],
    "dialogue": [
        "I want to learn machine learning but I don't know where to start.",
        "I've been feeling really stressed about work lately.",
        "I'm thinking about switching careers into tech.",
        "I just moved to a new city and don't know anyone.",
        "I want to start a side business but I have no idea what to do.",
    ],
    "empathy": [
        "I just got laid off after 8 years at the same company.",
        "My dog passed away this morning and I can't stop crying.",
        "I failed my exam again even though I studied so hard.",
        "I feel like nobody at work respects me.",
        "I've been dealing with anxiety and it's getting worse.",
    ],
    "conciseness": [
        "In order to be able to successfully complete the task that has been assigned to you, it is absolutely necessary and required that you first and foremost read through all of the documentation.",
        "Due to the fact that the weather conditions outside are not particularly favorable at this current point in time, it would probably be a good idea to consider the possibility of perhaps staying indoors.",
        "I wanted to reach out to you today for the purpose of letting you know that the meeting that was previously scheduled for tomorrow has been moved to a later date and time.",
        "It is my personal belief and opinion that the proposal that was submitted by the team last week should be reviewed and reconsidered before any final decisions are made.",
        "We are currently in the process of working on and developing a new feature that will hopefully allow users to be able to export their data in a more efficient manner.",
    ],
    "structure": [
        "First install python then you need pip and also make sure git is installed then clone the repo and run requirements and after that set up env variables.",
        "The benefits are it saves time and money and its easy to use and customers love it and it integrates with other tools and support is great.",
        "To bake the cake mix flour sugar butter eggs then bake at 350 and frost it after it cools and add sprinkles if you want.",
        "The report covers q1 revenue which was up 12 percent and customer growth was 8 percent and churn dropped and nps improved and we hired 20 people.",
        "For the onboarding process new users sign up then verify email then complete profile then watch intro video then connect integrations then invite team.",
    ],
    "writing": [
        "their going to the store and buyed alot of stuff for the party which its on saturday",
        "the meeting was went good and we discussed many things and decided to move forward with plan",
        "i think that maybe we should probably consider possibly looking into the option of maybe changing our approach",
        "he dont know nothing about the project and never did nothing to help nobody",
        "we was very excited about the new product launch and alot of customers are loving it so far",
    ],
    "router": [
        "I am so frustrated! I have been trying to cancel my subscription for weeks and nobody helps me!",
        "Can you explain quantum computing in simple terms?",
        "Write me a formal email to my boss asking for a raise.",
        "omg why is this so confusing, I dont get it at all lol",
        "I need step by step instructions to set up a VPN on my router.",
    ],
}




def call_api(model_id, system_prompt, user_message):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "max_tokens":  400,
        "temperature": 0.2,
    }
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        return None
    except Exception:
        return None


def call_finetuned(skill_name, system_prompt, user_message):
    merged_dir = os.path.join(FINETUNED_DIR, f"qwen25-0.5b-{skill_name}-v2-merged")
    if not os.path.exists(merged_dir):
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(merged_dir)
        model     = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            torch_dtype = torch.float16,
            device_map  = "auto",
        )
        model.eval()

        prompt = f"### System:\n{system_prompt}\n\n### User:\n{user_message}\n\n### Assistant:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = 400,
                temperature    = 0.2,
                do_sample      = True,
                pad_token_id   = tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        del model, tokenizer
        torch.cuda.empty_cache()

        return generated.strip()

    except Exception as e:
        print(f"    Finetuned inference error: {e}")
        return None


def judge_responses(skill_name, user_message, finetuned_response, big_model_response, big_model_name):
    judge_prompt = f"""You are an objective evaluator. Compare two AI responses to the same input for the skill: {skill_name}.

User message: "{user_message}"

Response A (Small fine-tuned 0.5B model):
{finetuned_response}

Response B ({big_model_name}):
{big_model_response}

Score each response from 1-10 on these criteria within the scope of {skill_name}:
1. accuracy: Is the core task completed correctly?
2. format: Is the output well-structured and appropriate?
3. relevance: Does it stay focused on the skill task?

Output ONLY this JSON:
{{
  "response_a": {{"accuracy": 0, "format": 0, "relevance": 0, "total": 0}},
  "response_b": {{"accuracy": 0, "format": 0, "relevance": 0, "total": 0}},
  "winner": "A|B|tie",
  "reason": "one sentence explanation"
}}"""

    result = call_api(JUDGE_MODEL, "You are an objective AI evaluator. Output only JSON.", judge_prompt)
    if not result:
        return None

    try:
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
            result = result.strip()
        return json.loads(result)
    except Exception:
        return None


def run_evaluation():
    all_results = {}

    for skill_name in SKILLS:
        print(f"\n{'='*60}")
        print(f"SKILL: {skill_name.upper()}")
        print(f"{'='*60}")

        config        = SKILL_PROMPTS[skill_name]
        system_prompt = config["system_prompt"]
        test_messages = TEST_CASES[skill_name]

        skill_results = {bm: [] for bm in BIG_MODELS}
        skill_results["summary"] = {}

        for i, user_message in enumerate(test_messages, 1):
            print(f"\n  [{i}/5] {user_message[:60]}...")

            finetuned_resp = call_finetuned(skill_name, system_prompt, user_message)
            if finetuned_resp is None:
                print(f"    Finetuned model not found or failed, skipping.")
                continue

            print(f"    Finetuned: {finetuned_resp[:80]}...")

            for bm_name, bm_id in BIG_MODELS.items():
                big_resp = call_api(bm_id, system_prompt, user_message)
                if big_resp is None:
                    print(f"    {bm_name}: API call failed.")
                    continue

                print(f"    {bm_name}: {big_resp[:80]}...")

                judgment = judge_responses(skill_name, user_message, finetuned_resp, big_resp, bm_name)
                if judgment is None:
                    print(f"    Judge failed.")
                    continue

                print(f"    Winner: {judgment.get('winner')} — {judgment.get('reason', '')[:60]}")

                skill_results[bm_name].append({
                    "user_message":     user_message,
                    "finetuned_resp":   finetuned_resp,
                    "big_model_resp":   big_resp,
                    "judgment":         judgment,
                })

        for bm_name in BIG_MODELS:
            comparisons = skill_results[bm_name]
            if not comparisons:
                continue

            wins_a   = sum(1 for c in comparisons if c["judgment"]["winner"] == "A")
            wins_b   = sum(1 for c in comparisons if c["judgment"]["winner"] == "B")
            ties     = sum(1 for c in comparisons if c["judgment"]["winner"] == "tie")
            avg_a    = sum(c["judgment"]["response_a"]["total"] for c in comparisons) / len(comparisons)
            avg_b    = sum(c["judgment"]["response_b"]["total"] for c in comparisons) / len(comparisons)

            skill_results["summary"][bm_name] = {
                "finetuned_wins": wins_a,
                "big_model_wins": wins_b,
                "ties":           ties,
                "avg_score_finetuned":  round(avg_a, 2),
                "avg_score_big_model":  round(avg_b, 2),
            }

            print(f"\n  vs {bm_name}:")
            print(f"    Finetuned wins : {wins_a}/5")
            print(f"    {bm_name} wins : {wins_b}/5")
            print(f"    Ties           : {ties}/5")
            print(f"    Avg score (finetuned): {avg_a:.1f}/30")
            print(f"    Avg score ({bm_name}): {avg_b:.1f}/30")

        all_results[skill_name] = skill_results

    return all_results


def print_final_summary(all_results):
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    for bm_name in BIG_MODELS:
        print(f"\nvs {bm_name}:")
        print(f"  {'SKILL':<25} {'FT WINS':<10} {'BIG WINS':<10} {'TIES':<6} {'FT AVG':<8} {'BIG AVG'}")
        print(f"  {'-'*75}")

        total_ft_wins  = 0
        total_bm_wins  = 0
        total_ties     = 0

        for skill_name in SKILLS:
            summary = all_results.get(skill_name, {}).get("summary", {}).get(bm_name)
            if not summary:
                print(f"  {skill_name:<25} {'N/A'}")
                continue

            total_ft_wins += summary["finetuned_wins"]
            total_bm_wins += summary["big_model_wins"]
            total_ties    += summary["ties"]

            print(f"  {skill_name:<25} {summary['finetuned_wins']:<10} {summary['big_model_wins']:<10} {summary['ties']:<6} {summary['avg_score_finetuned']:<8} {summary['avg_score_big_model']}")

        print(f"  {'-'*75}")
        print(f"  {'TOTAL':<25} {total_ft_wins:<10} {total_bm_wins:<10} {total_ties}")


if __name__ == "__main__":
    print("Starting skill evaluation...")
    print(f"Fine-tuned model dir: {FINETUNED_DIR}")
    print(f"Comparing against: {list(BIG_MODELS.keys())}")
    print(f"Judge: {JUDGE_MODEL}")

    all_results = run_evaluation()

    print_final_summary(all_results)

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nFull results saved to evaluation_results.json")