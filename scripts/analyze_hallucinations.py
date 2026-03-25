"""Analyze faithfulness results to categorize hallucination types."""
import json
import os

with open(os.path.join(os.path.dirname(__file__), "..", "evaluation-data", "faithfulness_scores.json"), encoding="utf-8") as f:
    d = json.load(f)

items = d["items"]

# --- Score distribution ---
print("=== SCORE DISTRIBUTION ===")
for s in range(1, 6):
    n = sum(1 for i in items if i.get("faithfulness_score") == s)
    print(f"  Score {s}: {n}")

# --- Total unsupported claims ---
all_claims = []
for i in items:
    all_claims.extend(i.get("unsupported_claims", []))
print(f"\nTotal individual unsupported claims: {len(all_claims)}")

# --- Categorize by pattern ---
print("\n=== HALLUCINATION CATEGORIES ===")
refusal = []
fabricated = []
overgeneralized = []
contradiction = []

for i in items:
    if not i.get("unsupported_claims") and i.get("faithfulness_score", 5) >= 4:
        continue
    r = i.get("reasoning", "").lower()
    q = i["id"]
    if any(x in r for x in ["failed to answer", "claims there is no", "deflection", "claims the context doesn"]):
        refusal.append(q)
    if any(x in r for x in ["fabricat", "not found in the", "not mentioned in the", "not in the source", "no basis in the", "not present in"]):
        fabricated.append(q)
    if any(x in r for x in ["overgeneraliz", "broader than", "goes beyond", "beyond what the source"]):
        overgeneralized.append(q)
    if any(x in r for x in ["contradict", "completely false", "incorrect", "opposite"]):
        contradiction.append(q)

print(f"  Refusal/deflection (had info but said it didn't): {len(refusal)}")
print(f"  Fabricated details (added stuff not in KB):       {len(fabricated)}")
print(f"  Overgeneralization (stretched the source):        {len(overgeneralized)}")
print(f"  Contradiction (stated the opposite):              {len(contradiction)}")

# --- Show examples of each type ---
print("\n=== EXAMPLE: REFUSAL/DEFLECTION (score 1) ===")
for qid in refusal[:3]:
    item = items[qid - 1]
    print(f"\nQ{item['id']}: {item['question']}")
    print(f"  Score: {item['faithfulness_score']}")
    print(f"  Reasoning: {item['reasoning'][:250]}")

print("\n=== EXAMPLE: FABRICATION (score 2-3) ===")
fab_items = [items[q-1] for q in fabricated if items[q-1].get("faithfulness_score") in (2, 3)]
for item in fab_items[:3]:
    print(f"\nQ{item['id']}: {item['question']}")
    print(f"  Score: {item['faithfulness_score']}")
    print(f"  Unsupported claims: {item['unsupported_claims'][:3]}")
    print(f"  Reasoning: {item['reasoning'][:250]}")

print("\n=== EXAMPLE: OVERGENERALIZATION (score 3-4) ===")
over_items = [items[q-1] for q in overgeneralized if items[q-1].get("faithfulness_score") in (3, 4)]
for item in over_items[:3]:
    print(f"\nQ{item['id']}: {item['question']}")
    print(f"  Score: {item['faithfulness_score']}")
    print(f"  Unsupported claims: {item['unsupported_claims'][:3]}")

print("\n=== EXAMPLE: CONTRADICTION (score 1-2) ===")
con_items = [items[q-1] for q in contradiction if items[q-1].get("faithfulness_score") in (1, 2)]
for item in con_items[:3]:
    print(f"\nQ{item['id']}: {item['question']}")
    print(f"  Score: {item['faithfulness_score']}")
    print(f"  Unsupported claims: {item['unsupported_claims'][:3]}")
    print(f"  Reasoning: {item['reasoning'][:250]}")
