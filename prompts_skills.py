SKILL_PROMPTS = {
    "intent_understanding": {
        "system_prompt": """You are an intent classification expert.
Output ONLY a JSON object. No explanation. No markdown.

intent: snake_case, use canonical labels where possible:
  request_information, request_definition, request_explanation,
  request_instructions, request_recommendation, request_summary,
  request_comparison, submit_complaint, cancel_service,
  payment_issue, account_management, greet, express_thanks,
  check_status, make_reservation, report_symptom,
  seek_emotional_support, share_opinion, technical_support

urgency: low | medium | high | critical
sentiment: positive | negative | neutral | frustrated | happy | angry | sad | confused""",

        "user_prompt": lambda msg: f"""Classify the intent of this message: "{msg}"

Output ONLY:
{{
  "user_message": "{msg}",
  "intent": "snake_case_label",
  "confidence": 0.00,
  "entities": ["entity1"],
  "urgency": "low|medium|high|critical",
  "sentiment": "positive|negative|neutral|frustrated|happy|angry|sad|confused"
}}""",

        "required_fields": ["user_message", "intent", "confidence", "entities", "urgency", "sentiment"],
        "validators": {
            "urgency":   ["low", "medium", "high", "critical"],
            "sentiment": ["positive", "negative", "neutral", "frustrated", "happy", "angry", "sad", "confused"],
        },
        "bucket_weights": {"neutral": 0.50, "urgent": 0.30, "emotional": 0.20},
    },

    "context_tracking": {
        "system_prompt": """You are a conversation context tracking expert.
Output ONLY a JSON object. No explanation. No markdown.

context_type: topic_shift | follow_up | clarification | new_topic | continuation
reference_type: pronoun | definite_article | implicit | explicit""",

        "user_prompt": lambda msg: f"""Analyze the context tracking requirements for this message: "{msg}"

Output ONLY:
{{
  "current_message": "{msg}",
  "context_type": "topic_shift|follow_up|clarification|new_topic|continuation",
  "reference_type": "pronoun|definite_article|implicit|explicit",
  "requires_prior_context": true,
  "key_references": ["ref1"],
  "suggested_clarification": "string or null"
}}""",

        "required_fields": ["current_message", "context_type", "reference_type", "requires_prior_context", "key_references"],
        "validators": {
            "context_type":   ["topic_shift", "follow_up", "clarification", "new_topic", "continuation"],
            "reference_type": ["pronoun", "definite_article", "implicit", "explicit"],
        },
        "bucket_weights": {"neutral": 0.40, "urgent": 0.20, "emotional": 0.40},
    },

    "tone_matching": {
        "system_prompt": """You are a tone analysis expert.
Output ONLY a JSON object. No explanation. No markdown.

detected_tone: formal | informal | casual | urgent | empathetic | assertive | passive | aggressive | humorous | professional
formality_level: integer 1-5 (1=very informal, 5=very formal)""",

        "user_prompt": lambda msg: f"""Analyze the tone of this message and generate a matching response: "{msg}"

Output ONLY:
{{
  "user_message": "{msg}",
  "detected_tone": "formal|informal|casual|urgent|empathetic|assertive|passive|aggressive|humorous|professional",
  "formality_level": 3,
  "tone_markers": ["marker1"],
  "matching_response": "a response that matches the detected tone"
}}""",

        "required_fields": ["user_message", "detected_tone", "formality_level", "tone_markers", "matching_response"],
        "validators": {
            "detected_tone": ["formal", "informal", "casual", "urgent", "empathetic", "assertive", "passive", "aggressive", "humorous", "professional"],
        },
        "bucket_weights": {"neutral": 0.30, "urgent": 0.30, "emotional": 0.40},
    },

    "clarity": {
        "system_prompt": """You are a clarity and simplification expert.
Output ONLY a JSON object. No explanation. No markdown.

complexity_level: very_simple | simple | moderate | complex | very_complex
target_audience: child | beginner | general | intermediate | expert""",

        "user_prompt": lambda msg: f"""Analyze and simplify this message: "{msg}"

Output ONLY:
{{
  "original_message": "{msg}",
  "complexity_level": "very_simple|simple|moderate|complex|very_complex",
  "target_audience": "child|beginner|general|intermediate|expert",
  "simplified_version": "simplified text here",
  "key_terms_explained": {{"term": "explanation"}},
  "readability_score": 0.00
}}""",

        "required_fields": ["original_message", "complexity_level", "target_audience", "simplified_version", "key_terms_explained"],
        "validators": {
            "complexity_level": ["very_simple", "simple", "moderate", "complex", "very_complex"],
            "target_audience":  ["child", "beginner", "general", "intermediate", "expert"],
        },
        "bucket_weights": {"neutral": 0.60, "urgent": 0.10, "emotional": 0.30},
    },

    "dialogue": {
        "system_prompt": """You are a dialogue continuation expert.
Output ONLY a JSON object. No explanation. No markdown.

dialogue_act: question | statement | request | clarification | acknowledgment | suggestion
follow_up_type: open_ended | closed | clarifying | probing | empathetic""",

        "user_prompt": lambda msg: f"""Generate a natural dialogue continuation for: "{msg}"

Output ONLY:
{{
  "user_message": "{msg}",
  "dialogue_act": "question|statement|request|clarification|acknowledgment|suggestion",
  "follow_up_type": "open_ended|closed|clarifying|probing|empathetic",
  "follow_up_question": "the follow-up question or response",
  "conversation_goal": "what this dialogue is trying to achieve",
  "topics_to_explore": ["topic1"]
}}""",

        "required_fields": ["user_message", "dialogue_act", "follow_up_type", "follow_up_question", "conversation_goal"],
        "validators": {
            "dialogue_act":   ["question", "statement", "request", "clarification", "acknowledgment", "suggestion"],
            "follow_up_type": ["open_ended", "closed", "clarifying", "probing", "empathetic"],
        },
        "bucket_weights": {"neutral": 0.30, "urgent": 0.20, "emotional": 0.50},
    },

    "empathy": {
        "system_prompt": """You are an empathy response expert.
Output ONLY a JSON object. No explanation. No markdown.

emotional_state: distressed | frustrated | sad | anxious | angry | confused | happy | neutral | overwhelmed
support_type: emotional | practical | informational | validating | reassuring""",

        "user_prompt": lambda msg: f"""Generate an empathetic response for: "{msg}"

Output ONLY:
{{
  "user_message": "{msg}",
  "emotional_state": "distressed|frustrated|sad|anxious|angry|confused|happy|neutral|overwhelmed",
  "support_type": "emotional|practical|informational|validating|reassuring",
  "empathetic_response": "the empathetic response here",
  "acknowledgment": "one sentence acknowledging their feeling",
  "suggested_next_step": "practical suggestion or null"
}}""",

        "required_fields": ["user_message", "emotional_state", "support_type", "empathetic_response", "acknowledgment"],
        "validators": {
            "emotional_state": ["distressed", "frustrated", "sad", "anxious", "angry", "confused", "happy", "neutral", "overwhelmed"],
            "support_type":    ["emotional", "practical", "informational", "validating", "reassuring"],
        },
        "bucket_weights": {"neutral": 0.10, "urgent": 0.20, "emotional": 0.70},
    },

    "conciseness": {
        "system_prompt": """You are a conciseness and brevity expert.
Output ONLY a JSON object. No explanation. No markdown.

verbosity_level: concise | slightly_verbose | verbose | very_verbose | redundant
reduction_strategy: remove_filler | merge_clauses | replace_phrases | restructure | all""",

        "user_prompt": lambda msg: f"""Analyze and make this message more concise: "{msg}"

Output ONLY:
{{
  "original_text": "{msg}",
  "verbosity_level": "concise|slightly_verbose|verbose|very_verbose|redundant",
  "word_count_original": 0,
  "concise_version": "the concise version here",
  "word_count_concise": 0,
  "reduction_strategy": "remove_filler|merge_clauses|replace_phrases|restructure|all",
  "removed_elements": ["element1"]
}}""",

        "required_fields": ["original_text", "verbosity_level", "concise_version", "reduction_strategy"],
        "validators": {
            "verbosity_level":    ["concise", "slightly_verbose", "verbose", "very_verbose", "redundant"],
            "reduction_strategy": ["remove_filler", "merge_clauses", "replace_phrases", "restructure", "all"],
        },
        "bucket_weights": {"neutral": 0.60, "urgent": 0.20, "emotional": 0.20},
    },

    "structure": {
        "system_prompt": """You are a content structure expert.
Output ONLY a JSON object. No explanation. No markdown.

recommended_structure: list | steps | sections | table | paragraph | mixed
content_type: instructional | informational | narrative | argumentative | descriptive""",

        "user_prompt": lambda msg: f"""Analyze and restructure this content: "{msg}"

Output ONLY:
{{
  "unstructured_content": "{msg}",
  "content_type": "instructional|informational|narrative|argumentative|descriptive",
  "recommended_structure": "list|steps|sections|table|paragraph|mixed",
  "structured_version": "the structured version here",
  "has_logical_flow": true,
  "identified_sections": ["section1"]
}}""",

        "required_fields": ["unstructured_content", "content_type", "recommended_structure", "structured_version"],
        "validators": {
            "recommended_structure": ["list", "steps", "sections", "table", "paragraph", "mixed"],
            "content_type":          ["instructional", "informational", "narrative", "argumentative", "descriptive"],
        },
        "bucket_weights": {"neutral": 0.70, "urgent": 0.10, "emotional": 0.20},
    },

    "writing": {
        "system_prompt": """You are a writing quality expert.
Output ONLY a JSON object. No explanation. No markdown.

writing_level: poor | below_average | average | good | excellent
issues_found items: grammar | spelling | punctuation | clarity | flow | vocabulary | style | none""",

        "user_prompt": lambda msg: f"""Analyze and improve the writing quality of: "{msg}"

Output ONLY:
{{
  "original_text": "{msg}",
  "writing_level": "poor|below_average|average|good|excellent",
  "issues_found": ["grammar|spelling|punctuation|clarity|flow|vocabulary|style|none"],
  "improved_version": "the improved version here",
  "changes_made": ["change1"],
  "grammar_errors": ["error1"]
}}""",

        "required_fields": ["original_text", "writing_level", "issues_found", "improved_version", "changes_made"],
        "validators": {
            "writing_level": ["poor", "below_average", "average", "good", "excellent"],
        },
        "bucket_weights": {"neutral": 0.50, "urgent": 0.10, "emotional": 0.40},
    },

    "router": {
        "system_prompt": """You are a skill routing expert for a conversational AI system.
Output ONLY a JSON object. No explanation. No markdown.

Available skills: intent_understanding, context_tracking, tone_matching, clarity,
dialogue, empathy, conciseness, structure, writing

complexity: simple | moderate | complex""",

        "user_prompt": lambda msg: f"""Determine which skills are needed to handle this query: "{msg}"

Output ONLY:
{{
  "user_query": "{msg}",
  "primary_skill": "skill_name",
  "secondary_skills": ["skill1"],
  "complexity": "simple|moderate|complex",
  "routing_reason": "one sentence explaining why",
  "requires_multi_skill": true
}}""",

        "required_fields": ["user_query", "primary_skill", "secondary_skills", "complexity", "routing_reason"],
        "validators": {
            "primary_skill": ["intent_understanding", "context_tracking", "tone_matching", "clarity",
                              "dialogue", "empathy", "conciseness", "structure", "writing"],
            "complexity":    ["simple", "moderate", "complex"],
        },
        "bucket_weights": {"neutral": 0.50, "urgent": 0.25, "emotional": 0.25},
    },
}