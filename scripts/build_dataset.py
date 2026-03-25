"""
Dataset Builder for FAQ Quality Evaluation
Downloads the FAQ knowledge base and queries the live chatbot for each question.
Outputs a structured dataset.json for use in expert review and automated metrics.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

KB_URL = "https://storupstream.blob.core.windows.net/$web/site/supportcenter/support.json"
DEFAULT_ENDPOINT = "https://upstreamaifaq-uat.azurewebsites.net/api/faqquery?code=<REDACTED>"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "evaluation-data", "dataset.json")


def download_kb(url: str) -> list[dict]:
    print(f"Downloading knowledge base from {url}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(f"  -> {len(data)} Q&A pairs loaded.")
    return data


def query_chatbot(endpoint: str, question: str) -> dict:
    payload = {"faqquery": question}
    resp = requests.post(endpoint, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def load_existing_dataset(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_dataset(path: str, dataset: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Build evaluation dataset from FAQ KB + chatbot")
    parser.add_argument("--endpoint", default=None, help="Chatbot API endpoint URL")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output file path")
    args = parser.parse_args()

    endpoint = args.endpoint or os.environ.get("CHATBOT_ENDPOINT", DEFAULT_ENDPOINT)
    print(f"Using chatbot endpoint: {endpoint}")

    # Download KB
    kb = download_kb(KB_URL)

    # Load existing dataset for resume capability
    existing = load_existing_dataset(args.output)
    answered_questions = set()
    items = []
    if existing and "items" in existing:
        items = existing["items"]
        answered_questions = {item["question"] for item in items}
        print(f"Resuming: {len(answered_questions)} questions already answered.")

    # Query chatbot for each KB question
    total = len(kb)
    for i, qa in enumerate(kb):
        question = qa.get("question", "")
        ground_truth = qa.get("answer", "")

        if question in answered_questions:
            continue

        item_id = len(items) + 1
        print(f"[{item_id}/{total}] Querying: {question[:80]}...")

        try:
            result = query_chatbot(endpoint, question)
            chatbot_response = result.get("response", "")
            elapsed = result.get("elapsed_time", None)
        except requests.RequestException as e:
            print(f"  ERROR: {e}")
            chatbot_response = f"ERROR: {e}"
            elapsed = None

        items.append({
            "id": item_id,
            "question": question,
            "ground_truth": ground_truth,
            "chatbot_response": chatbot_response,
            "elapsed_time_seconds": elapsed,
        })

        # Save after each question for crash resilience
        dataset = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "endpoint": endpoint,
                "source_url": KB_URL,
                "total_questions": total,
                "completed_questions": len(items),
            },
            "items": items,
        }
        save_dataset(args.output, dataset)

        if i < total - 1:
            time.sleep(args.delay)

    print(f"\nDone. {len(items)} items saved to {args.output}")


if __name__ == "__main__":
    main()
