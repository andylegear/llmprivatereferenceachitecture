"""
Compute faithfulness scores using Claude as a judge.
For each chatbot response, asks Claude whether the answer is grounded
in the source material (ground truth).
Requires ANTHROPIC_API_KEY environment variable.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "..", "evaluation-data", "dataset.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "..", "evaluation-data", "faithfulness_scores.json")

MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


JUDGE_PROMPT = """You are an expert evaluator assessing the faithfulness of a chatbot's response to a user question about an FAQ system.

**Faithfulness** means: every claim in the chatbot's response is directly supported by the source material (ground truth answer). The response should not contain fabricated information, unsupported claims, or hallucinations.

**Important context — exclusions from hallucination scoring**:
1. **Contact info boilerplate**: The chatbot's system prompt and application code inject references to "support@upstream.exchange" and "https://upstream.exchange/SupportCenter". These are intentional system-level additions. Do NOT count them as unsupported claims.
2. **Overgeneralization**: If the chatbot broadens, paraphrases, or slightly stretches a claim from the source material without inventing entirely new information, this is NOT a hallucination. For example, summarising "wire transfer of funds in the U.S. and internationally" as "international wire transfers" is acceptable. Only count a claim as unsupported if it introduces substantively new information with no basis in the source.
3. **Refusal to answer**: If the chatbot declines to answer or says it does not have the information, this is a retrieval failure, NOT a hallucination. A refusal contains no fabricated claims and should be scored based only on whatever affirmative claims it does make. A pure refusal with no affirmative claims should score 5 (no unsupported claims were made).

## Source Material (Ground Truth)
{ground_truth}

## User Question
{question}

## Chatbot Response
{chatbot_response}

## Your Task
1. Rate the faithfulness of the chatbot's response on a scale of 1-5:
   1 = Completely unfaithful (mostly fabricated information)
   2 = Mostly unfaithful (significant unsupported claims)
   3 = Partially faithful (mix of supported and unsupported claims)
   4 = Mostly faithful (minor unsupported details)
   5 = Fully faithful (all claims supported by source)

2. List any specific claims in the response that are NOT supported by the source material. Exclude contact-info boilerplate, overgeneralizations, and pure refusals from this list.

3. Provide brief reasoning for your score.

Respond in this exact JSON format:
{{
  "faithfulness_score": <1-5>,
  "unsupported_claims": ["claim 1", "claim 2"],
  "reasoning": "brief explanation"
}}"""


def load_dataset(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_results(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
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
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            return json.loads(text)
        except json.JSONDecodeError:
            # If Claude didn't return valid JSON, return a structured error
            return {
                "faithfulness_score": None,
                "unsupported_claims": [],
                "reasoning": f"Failed to parse Claude response: {text[:200]}",
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    Retry {attempt + 1}/{MAX_RETRIES} after error: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return {
                    "faithfulness_score": None,
                    "unsupported_claims": [],
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

    # Resume capability
    existing = load_existing_results(OUTPUT_PATH)
    scored_ids = set()
    result_items = []
    if existing and "items" in existing:
        result_items = existing["items"]
        scored_ids = {item["id"] for item in result_items if item.get("faithfulness_score") is not None}
        print(f"Resuming: {len(scored_ids)} items already scored.")

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
        output = {
            "metadata": {
                "dataset": DATASET_PATH,
                "model": MODEL,
                "computed_at": datetime.now(timezone.utc).isoformat(),
            },
            "items": result_items,
        }
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # Brief pause to respect rate limits
        time.sleep(0.5)

    # Compute summary
    scored = [item for item in result_items if item.get("faithfulness_score") is not None]
    if scored:
        scores = [item["faithfulness_score"] for item in scored]
        hall_count = sum(1 for item in scored if item.get("unsupported_claims"))
        summary = {
            "total_scored": len(scored),
            "faithfulness_mean": round(sum(scores) / len(scores), 2),
            "faithfulness_median": round(sorted(scores)[len(scores) // 2], 2),
            "faithfulness_min": min(scores),
            "faithfulness_max": max(scores),
            "items_with_hallucinations": hall_count,
        }
    else:
        summary = {"total_scored": 0}

    output = {
        "metadata": {
            "dataset": DATASET_PATH,
            "model": MODEL,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "items": result_items,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Results saved to {OUTPUT_PATH}")
    print(f"{'='*50}")
    if scored:
        print(f"  Items scored:              {summary['total_scored']}")
        print(f"  Faithfulness mean:         {summary['faithfulness_mean']}")
        print(f"  Faithfulness median:       {summary['faithfulness_median']}")
        print(f"  Faithfulness range:        {summary['faithfulness_min']}-{summary['faithfulness_max']}")
        print(f"  Items w/ hallucinations:   {summary['items_with_hallucinations']}")


if __name__ == "__main__":
    main()
