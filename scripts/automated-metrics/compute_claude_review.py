"""
Claude-as-judge evaluation using the same rubric as human expert reviewers.
Scores each chatbot response on: Accuracy (1-5), Completeness (1-5),
Helpfulness (1-5), and Hallucination (yes/no).

This enables inter-rater agreement analysis between Claude and human reviewers.
Requires ANTHROPIC_API_KEY environment variable.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "..", "evaluation-data", "dataset.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "..", "evaluation-data", "claude_review_scores.json")

MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
MAX_TOKENS = 1024


JUDGE_PROMPT = """You are a domain expert evaluating an FAQ chatbot that answers questions about the Upstream trading platform. You are scoring the chatbot's response using the exact same rubric as human expert reviewers.

**Important context — exclusions**:
1. **Contact info boilerplate**: Every response includes a system-injected footer directing users to "support@upstream.exchange" and "https://upstream.exchange/SupportCenter". Ignore this boilerplate entirely — do not let it affect any score.
2. **Overgeneralization**: If the chatbot paraphrases or slightly broadens a claim without inventing new information, do not penalise it. Only flag substantively new claims with no basis in the source.
3. **Refusal to answer**: If the chatbot says it cannot answer, score Accuracy based only on any affirmative claims made. Score Completeness low (it missed the content). Score Helpfulness low (it didn't help the user). Only flag Hallucination if it fabricated specific claims.

## Source Material (Ground Truth)
{ground_truth}

## User Question
{question}

## Chatbot Response
{chatbot_response}

## Scoring Rubric

Rate the response on these four dimensions:

### 1. Accuracy (1-5): Is the response factually correct?
1 = Completely incorrect
2 = Mostly incorrect
3 = Partially correct
4 = Mostly correct
5 = Fully correct

### 2. Completeness (1-5): Does it cover all relevant information from the source?
1 = Missing everything
2 = Major gaps
3 = Covers some key points
4 = Covers most
5 = Fully complete

### 3. Helpfulness (1-5): Would this adequately help a real user?
1 = Not helpful at all
2 = Slightly helpful
3 = Moderately helpful
4 = Very helpful
5 = Extremely helpful

### 4. Hallucination: Does the response contain information NOT in the source material?
true = Yes — contains unsupported or fabricated information
false = No — all claims are supported by source (or it is a pure refusal)

You MUST respond in this exact JSON format with ALL five fields present. The "hallucination" field is REQUIRED and must be a boolean (true or false), never omit it:
{{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "helpfulness": <1-5>,
  "hallucination": <true or false>,
  "reasoning": "brief explanation including whether hallucination was detected"
}}

IMPORTANT: You must always include the "hallucination" field with value true or false."""


def load_dataset(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_results(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data and "items" in data:
                return data
    return None


def judge_item(client, question: str, ground_truth: str, chatbot_response: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        ground_truth=ground_truth,
        question=question,
        chatbot_response=chatbot_response,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            result = json.loads(text)

            # Validate expected fields
            for field in ("accuracy", "completeness", "helpfulness"):
                if field not in result or not isinstance(result[field], int):
                    result[field] = None

            # Coerce hallucination to bool (handle string "true"/"false")
            hall = result.get("hallucination")
            if isinstance(hall, str):
                result["hallucination"] = hall.lower() == "true"
            elif isinstance(hall, bool):
                pass  # already correct
            else:
                # Hallucination missing — try to infer from reasoning
                reasoning = result.get("reasoning", "").lower()
                if "no hallucination" in reasoning or "no fabricat" in reasoning or "all claims are supported" in reasoning:
                    result["hallucination"] = False
                elif "hallucination" in reasoning and ("contains" in reasoning or "fabricat" in reasoning or "unsupported" in reasoning):
                    result["hallucination"] = True
                else:
                    # Retry if we have attempts left
                    if attempt < MAX_RETRIES - 1:
                        print(f"    Retry {attempt + 1}/{MAX_RETRIES}: hallucination field missing")
                        time.sleep(RETRY_DELAY)
                        continue
                    result["hallucination"] = None

            return result
        except json.JSONDecodeError:
            if attempt < MAX_RETRIES - 1:
                print(f"    Retry {attempt + 1}/{MAX_RETRIES} after JSON parse error")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return {
                "accuracy": None,
                "completeness": None,
                "helpfulness": None,
                "hallucination": None,
                "reasoning": f"Failed to parse Claude response after {MAX_RETRIES} attempts: {text[:200]}",
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    Retry {attempt + 1}/{MAX_RETRIES} after error: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return {
                    "accuracy": None,
                    "completeness": None,
                    "helpfulness": None,
                    "hallucination": None,
                    "reasoning": f"API error after {MAX_RETRIES} retries: {e}",
                }


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Run setup_env.ps1 first, or set it manually.")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed.")
        print("Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    items = dataset["items"]

    # Resume capability — re-score items with null hallucination or null accuracy
    existing = load_existing_results(OUTPUT_PATH)
    scored_ids = set()
    result_items = []
    rescore_ids = set()
    if existing:
        for item in existing["items"]:
            if (item.get("accuracy") is not None
                    and item.get("hallucination") is not None):
                scored_ids.add(item["id"])
                result_items.append(item)
            else:
                rescore_ids.add(item["id"])
        print(f"Resuming: {len(scored_ids)} fully scored, {len(rescore_ids)} to re-score.")

    remaining = [item for item in items if item["id"] not in scored_ids
                 and not item["chatbot_response"].startswith("ERROR:")]
    print(f"Scoring {len(remaining)} items with Claude ({MODEL})...\n")

    for i, item in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] Item {item['id']}: {item['question'][:60]}...")

        result = judge_item(
            client,
            item["question"],
            item["ground_truth"],
            item["chatbot_response"],
        )

        result_items.append({
            "id": item["id"],
            "question": item["question"],
            **result,
        })

        # Save after each item for crash resilience
        save_results(result_items, dataset)

        # Brief pause to respect rate limits
        time.sleep(0.5)

    # Final save with summary
    save_results(result_items, dataset, final=True)

    # Print summary
    scored = [item for item in result_items if item.get("accuracy") is not None]
    if scored:
        for dim in ("accuracy", "completeness", "helpfulness"):
            vals = [item[dim] for item in scored if item.get(dim)]
            mean = sum(vals) / len(vals) if vals else 0
            print(f"  {dim.capitalize()} mean:        {mean:.2f}")
        hall = sum(1 for item in scored if item.get("hallucination"))
        print(f"  Hallucinations flagged:    {hall}/{len(scored)}")


def save_results(result_items, dataset, final=False):
    scored = [item for item in result_items if item.get("accuracy") is not None]

    summary = {}
    if scored:
        for dim in ("accuracy", "completeness", "helpfulness"):
            vals = [item[dim] for item in scored if item.get(dim)]
            if vals:
                sorted_vals = sorted(vals)
                summary[f"{dim}_mean"] = round(sum(vals) / len(vals), 2)
                summary[f"{dim}_median"] = sorted_vals[len(sorted_vals) // 2]
        summary["total_scored"] = len(scored)
        summary["hallucinations_flagged"] = sum(1 for item in scored if item.get("hallucination"))

    output = {
        "metadata": {
            "dataset": DATASET_PATH,
            "model": MODEL,
            "rubric": "Expert review rubric (Accuracy, Completeness, Helpfulness 1-5 + Hallucination flag)",
            "computed_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "items": result_items,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if final:
        print(f"\n{'='*50}")
        print(f"Results saved to {OUTPUT_PATH}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
