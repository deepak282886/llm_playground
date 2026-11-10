import os
API_KEY = os.getenv("TOGETHER_API_KEY", "YOUR_API_KEY_HERE")
API_URL          = "https://api.together.xyz/v1/chat/completions"
MODEL            = "openai/gpt-oss-20b"
TARGET_EXAMPLES  = 5000
PARALLEL_WORKERS = 20
MAX_TOKENS       = 400
OUTPUT_DIR       = "train_data"
# SKILLS = [
#     "intent_understanding",
#     "context_tracking",
#     "tone_matching",
#     "clarity",
#     "dialogue",
#     "empathy",
#     "conciseness",
#     "structure",
#     "writing",
#     "router",
# ]

SKILLS = [
    # "intent_understanding",
    "router"
]