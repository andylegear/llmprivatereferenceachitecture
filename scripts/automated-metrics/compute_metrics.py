"""
Compute ROUGE and BERTScore metrics for chatbot responses vs ground truth.
Reads dataset.json and outputs automated_metrics.json to results/.
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "..", "evaluation-data", "dataset.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "..", "evaluation-data", "automated_metrics.json")


def load_dataset(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_rouge_scores(references: list[str], hypotheses: list[str]) -> list[dict]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = []
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        results.append({
            "rouge1_f1": round(scores["rouge1"].fmeasure, 4),
            "rouge2_f1": round(scores["rouge2"].fmeasure, 4),
            "rougeL_f1": round(scores["rougeL"].fmeasure, 4),
        })
    return results


def compute_bert_scores(references: list[str], hypotheses: list[str]) -> list[dict]:
    from bert_score import score as bert_score

    P, R, F1 = bert_score(hypotheses, references, model_type="distilbert-base-uncased", verbose=True)
    results = []
    for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist()):
        results.append({
            "bertscore_precision": round(p, 4),
            "bertscore_recall": round(r, 4),
            "bertscore_f1": round(f, 4),
        })
    return results


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    items = dataset["items"]

    references = [item["ground_truth"] for item in items]
    hypotheses = [item["chatbot_response"] for item in items]

    # Filter out error responses
    valid = [(r, h, item) for r, h, item in zip(references, hypotheses, items)
             if not h.startswith("ERROR:")]
    if len(valid) < len(items):
        print(f"  Skipping {len(items) - len(valid)} items with error responses.")
    references, hypotheses, valid_items = zip(*valid) if valid else ([], [], [])
    references, hypotheses = list(references), list(hypotheses)

    print(f"\nComputing ROUGE scores for {len(references)} items...")
    rouge_results = compute_rouge_scores(references, hypotheses)

    print(f"\nComputing BERTScore for {len(references)} items...")
    bert_results = compute_bert_scores(references, hypotheses)

    # Merge results
    output_items = []
    for item, rouge, bert in zip(valid_items, rouge_results, bert_results):
        output_items.append({
            "id": item["id"],
            "question": item["question"],
            **rouge,
            **bert,
        })

    # Compute aggregates
    def mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0

    summary = {
        "total_items": len(output_items),
        "rouge1_f1_mean": mean([i["rouge1_f1"] for i in output_items]),
        "rouge2_f1_mean": mean([i["rouge2_f1"] for i in output_items]),
        "rougeL_f1_mean": mean([i["rougeL_f1"] for i in output_items]),
        "bertscore_f1_mean": mean([i["bertscore_f1"] for i in output_items]),
        "bertscore_precision_mean": mean([i["bertscore_precision"] for i in output_items]),
        "bertscore_recall_mean": mean([i["bertscore_recall"] for i in output_items]),
    }

    output = {
        "metadata": {
            "dataset": DATASET_PATH,
            "computed_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        },
        "summary": summary,
        "items": output_items,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Results saved to {OUTPUT_PATH}")
    print(f"{'='*50}")
    print(f"  Items evaluated:        {summary['total_items']}")
    print(f"  ROUGE-1 F1 (mean):      {summary['rouge1_f1_mean']}")
    print(f"  ROUGE-2 F1 (mean):      {summary['rouge2_f1_mean']}")
    print(f"  ROUGE-L F1 (mean):      {summary['rougeL_f1_mean']}")
    print(f"  BERTScore F1 (mean):    {summary['bertscore_f1_mean']}")
    print(f"  BERTScore Prec. (mean): {summary['bertscore_precision_mean']}")
    print(f"  BERTScore Rec. (mean):  {summary['bertscore_recall_mean']}")


if __name__ == "__main__":
    main()
