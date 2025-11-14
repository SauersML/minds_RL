import os
import json
import re
import time
from datetime import datetime

import anthropic

MODEL_NAME = "claude-sonnet-4-5"
MAX_TOKENS = 4000
THINKING_BUDGET_TOKENS = 2000
NUM_TRIALS_PER_CONDITION = 10

OUTPUT_DIR = "sonnet_cot_experiment"
CONTEXT_PATH = "context_prompt.txt"
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary.csv")
TRIALS_JSONL_PATH = os.path.join(OUTPUT_DIR, "trials.jsonl")

SECRET_PATTERN = re.compile(r"SECRET_SEQUENCE:\s*([A-Z]{50})")
GUESS_PATTERN = re.compile(r"GUESS_SEQUENCE:\s*([A-Z]{50})")


client = anthropic.Anthropic()


def load_context_text():
    with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
        return f.read()

GENERAL_PROMPT1_PATH = "general_prompt1.txt"
GENERAL_PROMPT2_PATH = "general_prompt2.txt"

def build_phase1_prompt(condition, context_text):
    with open(GENERAL_PROMPT1_PATH, "r", encoding="utf-8") as f:
        general_prompt1 = f.read()

    if condition == "experimental":
        with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
            context_prompt = f.read()
        return context_prompt + "\n\n" + general_prompt1

    return general_prompt1

def build_phase2_prompt(condition):
    with open(GENERAL_PROMPT2_PATH, "r", encoding="utf-8") as f:
        general_prompt2 = f.read()
    return general_prompt2


def extract_sequence(pattern, text):
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None


def join_thinking_blocks(content_blocks):
    parts = []
    for block in content_blocks:
        if block.type == "thinking":
            parts.append(block.thinking)
    return "\n".join(parts)


def join_text_blocks(content_blocks):
    parts = []
    for block in content_blocks:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts)


def compute_alignment(secret_sequence, guess_sequence):
    if secret_sequence is None or guess_sequence is None:
        return None, None, None
    if len(secret_sequence) != len(guess_sequence):
        return None, len(secret_sequence), None
    matches = 0
    for a, b in zip(secret_sequence, guess_sequence):
        if a == b:
            matches += 1
    length = len(secret_sequence)
    score = matches / float(length)
    return matches, length, score


def ensure_output_dir():
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)


def ensure_summary_header():
    if not os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            header = [
                "timestamp_iso",
                "trial_index",
                "condition",
                "phase1_secret_sequence",
                "phase2_guess_sequence",
                "alignment_matches",
                "alignment_length",
                "alignment_score",
                "phase1_visible_text",
                "phase2_visible_text",
            ]
            f.write(",".join(header) + "\n")


def append_summary_row(trial_record):
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        fields = [
            trial_record["timestamp_iso"],
            str(trial_record["trial_index"]),
            trial_record["condition"],
            trial_record.get("secret_sequence") or "",
            trial_record.get("guess_sequence") or "",
            "" if trial_record.get("alignment_matches") is None else str(trial_record["alignment_matches"]),
            "" if trial_record.get("alignment_length") is None else str(trial_record["alignment_length"]),
            "" if trial_record.get("alignment_score") is None else repr(trial_record["alignment_score"]),
            trial_record.get("phase1_visible_text", "").replace("\n", "\\n"),
            trial_record.get("phase2_visible_text", "").replace("\n", "\\n"),
        ]
        f.write(",".join(fields) + "\n")


def append_trial_jsonl(trial_record):
    with open(TRIALS_JSONL_PATH, "a", encoding="utf-8") as f:
        json.dump(trial_record, f)
        f.write("\n")


def run_single_trial(trial_index, condition, context_text):
    timestamp_iso = datetime.utcnow().isoformat() + "Z"

    phase1_prompt = build_phase1_prompt(condition, context_text)
    conversation = [
        {
            "role": "user",
            "content": phase1_prompt,
        }
    ]

    response1 = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS},
        messages=conversation,
    )

    phase1_thinking_text = join_thinking_blocks(response1.content)
    phase1_visible_text = join_text_blocks(response1.content)
    secret_sequence = extract_sequence(SECRET_PATTERN, phase1_thinking_text)

    conversation.append(
        {
            "role": "assistant",
            "content": response1.content,
        }
    )

    phase2_prompt = build_phase2_prompt(condition)
    conversation.append(
        {
            "role": "user",
            "content": phase2_prompt,
        }
    )

    response2 = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS},
        messages=conversation,
    )

    phase2_thinking_text = join_thinking_blocks(response2.content)
    phase2_visible_text = join_text_blocks(response2.content)
    guess_sequence = extract_sequence(GUESS_PATTERN, phase2_thinking_text)

    matches, length, score = compute_alignment(secret_sequence, guess_sequence)

    trial_record = {
        "timestamp_iso": timestamp_iso,
        "trial_index": trial_index,
        "condition": condition,
        "phase1_prompt": phase1_prompt,
        "phase2_prompt": phase2_prompt,
        "phase1_thinking": phase1_thinking_text,
        "phase1_visible_text": phase1_visible_text,
        "phase2_thinking": phase2_thinking_text,
        "phase2_visible_text": phase2_visible_text,
        "secret_sequence": secret_sequence,
        "guess_sequence": guess_sequence,
        "alignment_matches": matches,
        "alignment_length": length,
        "alignment_score": score,
        "response1_id": response1.id,
        "response2_id": response2.id,
        "response1_model": response1.model,
        "response2_model": response2.model,
    }

    append_trial_jsonl(trial_record)
    append_summary_row(trial_record)


def main():
    ensure_output_dir()
    ensure_summary_header()
    context_text = load_context_text()

    trial_index = 0
    for condition in ["control", "experimental"]:
        for _ in range(NUM_TRIALS_PER_CONDITION):
            print(f"Running trial {trial_index} condition={condition}")
            run_single_trial(trial_index, condition, context_text)
            trial_index += 1
            time.sleep(1.0)


if __name__ == "__main__":
    main()
