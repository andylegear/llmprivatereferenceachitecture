"""
Inter-rater agreement analysis: Human expert reviewers vs Claude reviewer.

Computes:
- Cohen's kappa (pairwise, per dimension)
- Krippendorff's alpha (all raters, per dimension)
- Correlation and mean differences
- Hallucination agreement (confusion matrix)
- Per-dimension distribution comparison

Automatically discovers all evaluation-*.json files in results/.
Re-run whenever new expert reviews are added.

Requires: scikit-learn, krippendorff, numpy, scipy
"""

import glob
import json
import os
import sys
from collections import Counter

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "evaluation-data")
CLAUDE_PATH = os.path.join(RESULTS_DIR, "claude_review_scores.json")

DIMENSIONS = ["accuracy", "completeness", "helpfulness"]


def load_claude_reviews(path: str) -> dict:
    """Load Claude review scores, keyed by item ID."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data["items"]}


def load_human_reviews(results_dir: str) -> list[dict]:
    """Discover and load all evaluation-*.json files."""
    pattern = os.path.join(results_dir, "evaluation-*.json")
    files = sorted(glob.glob(pattern))
    reviews = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        reviewer_id = data.get("reviewer_id", os.path.basename(fpath))
        ratings_by_id = {r["item_id"]: r for r in data["ratings"]}
        reviews.append({
            "reviewer_id": reviewer_id,
            "file": os.path.basename(fpath),
            "ratings": ratings_by_id,
            "sample_size": data.get("sample_size", len(data["ratings"])),
        })
    return reviews


def cohens_kappa_ordinal(y1, y2):
    """Cohen's kappa with linear weights for ordinal scales."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y1, y2, weights="linear")


def cohens_kappa_nominal(y1, y2):
    """Cohen's kappa for nominal/binary data."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y1, y2)


def krippendorff_alpha(reliability_data, level="ordinal"):
    """Krippendorff's alpha for a reliability data matrix."""
    import krippendorff as ka
    return ka.alpha(reliability_data=reliability_data, level_of_measurement=level)


def get_overlapping_scores(claude: dict, human: dict, dimension: str):
    """Get paired scores for overlapping items."""
    claude_scores = []
    human_scores = []
    for item_id, human_rating in human.items():
        if item_id in claude:
            c_val = claude[item_id].get(dimension)
            h_val = human_rating.get(dimension)
            if c_val is not None and h_val is not None:
                claude_scores.append(c_val)
                human_scores.append(h_val)
    return claude_scores, human_scores


def get_overlapping_hallucination(claude: dict, human: dict):
    """Get paired hallucination flags for overlapping items."""
    claude_flags = []
    human_flags = []
    for item_id, human_rating in human.items():
        if item_id in claude:
            c_val = claude[item_id].get("hallucination")
            h_val = human_rating.get("hallucination")
            if c_val is not None and h_val is not None:
                claude_flags.append(c_val)
                human_flags.append(h_val)
    return claude_flags, human_flags


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def analyze_pairwise(claude: dict, human_review: dict):
    """Full pairwise analysis between Claude and one human reviewer."""
    reviewer_id = human_review["reviewer_id"]
    ratings = human_review["ratings"]
    overlap_count = sum(1 for iid in ratings if iid in claude)

    print_section(f"Claude vs {reviewer_id} ({overlap_count} overlapping items)")

    # --- Ordinal dimensions ---
    print(f"\n  {'Dimension':<16} {'Kappa_w':>8} {'r':>8} {'Claude_μ':>9} {'Human_μ':>8} {'Δμ':>7} {'MAE':>6}")
    print(f"  {'-'*14:<16} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*7} {'-'*6}")

    from scipy.stats import pearsonr, spearmanr

    all_kappas = {}
    for dim in DIMENSIONS:
        c_scores, h_scores = get_overlapping_scores(claude, ratings, dim)
        if len(c_scores) < 2:
            print(f"  {dim:<16} insufficient data")
            continue

        c_arr = np.array(c_scores)
        h_arr = np.array(h_scores)

        kappa = cohens_kappa_ordinal(c_scores, h_scores)
        r_val, p_val = pearsonr(c_arr, h_arr)
        rho, _ = spearmanr(c_arr, h_arr)
        c_mean = c_arr.mean()
        h_mean = h_arr.mean()
        delta = c_mean - h_mean
        mae = np.abs(c_arr - h_arr).mean()

        all_kappas[dim] = kappa

        print(f"  {dim:<16} {kappa:>8.3f} {r_val:>8.3f} {c_mean:>9.2f} {h_mean:>8.2f} {delta:>+7.2f} {mae:>6.2f}")

    # --- Hallucination agreement ---
    c_flags, h_flags = get_overlapping_hallucination(claude, ratings)
    if len(c_flags) >= 2:
        kappa_hall = cohens_kappa_nominal(
            [int(x) for x in c_flags],
            [int(x) for x in h_flags],
        )
        all_kappas["hallucination"] = kappa_hall

        # Confusion matrix
        tp = sum(1 for c, h in zip(c_flags, h_flags) if c and h)  # both say hallucination
        tn = sum(1 for c, h in zip(c_flags, h_flags) if not c and not h)  # both say no
        fp = sum(1 for c, h in zip(c_flags, h_flags) if c and not h)  # Claude says yes, human no
        fn = sum(1 for c, h in zip(c_flags, h_flags) if not c and h)  # Claude says no, human yes

        c_rate = sum(c_flags) / len(c_flags) * 100
        h_rate = sum(h_flags) / len(h_flags) * 100
        agreement = (tp + tn) / len(c_flags) * 100

        print(f"\n  Hallucination:")
        print(f"    Cohen's kappa:     {kappa_hall:.3f}")
        print(f"    Raw agreement:     {agreement:.1f}%")
        print(f"    Claude flag rate:  {sum(c_flags)}/{len(c_flags)} ({c_rate:.1f}%)")
        print(f"    Human flag rate:   {sum(h_flags)}/{len(h_flags)} ({h_rate:.1f}%)")
        print(f"\n    Confusion matrix (Claude \\ Human):")
        print(f"                       Human=No  Human=Yes")
        print(f"      Claude=No        {tn:>5}      {fn:>5}")
        print(f"      Claude=Yes       {fp:>5}      {tp:>5}")

    # --- Score distribution comparison ---
    print(f"\n  Score distributions (Claude / Human):")
    for dim in DIMENSIONS:
        c_scores, h_scores = get_overlapping_scores(claude, ratings, dim)
        c_dist = Counter(c_scores)
        h_dist = Counter(h_scores)
        print(f"    {dim}:")
        for s in range(1, 6):
            c_n = c_dist.get(s, 0)
            h_n = h_dist.get(s, 0)
            c_bar = "█" * c_n
            h_bar = "▒" * h_n
            print(f"      {s}: C={c_n:>2} {c_bar}")
            print(f"         H={h_n:>2} {h_bar}")

    # --- Exact and adjacent agreement ---
    print(f"\n  Agreement rates:")
    print(f"  {'Dimension':<16} {'Exact':>7} {'±1':>7} {'±2':>7}")
    print(f"  {'-'*14:<16} {'-'*7} {'-'*7} {'-'*7}")
    for dim in DIMENSIONS:
        c_scores, h_scores = get_overlapping_scores(claude, ratings, dim)
        if not c_scores:
            continue
        n = len(c_scores)
        exact = sum(1 for c, h in zip(c_scores, h_scores) if c == h) / n * 100
        adj1 = sum(1 for c, h in zip(c_scores, h_scores) if abs(c - h) <= 1) / n * 100
        adj2 = sum(1 for c, h in zip(c_scores, h_scores) if abs(c - h) <= 2) / n * 100
        print(f"  {dim:<16} {exact:>6.1f}% {adj1:>6.1f}% {adj2:>6.1f}%")

    return all_kappas


def analyze_krippendorff(claude: dict, human_reviews: list[dict]):
    """Compute Krippendorff's alpha across all raters (Claude + all humans)."""
    print_section("Krippendorff's Alpha (all raters: Claude + humans)")

    # Collect all unique item IDs that at least 2 raters scored
    all_rater_ids = ["Claude"] + [r["reviewer_id"] for r in human_reviews]
    rater_data = [claude] + [r["ratings"] for r in human_reviews]

    for dim in DIMENSIONS + ["hallucination"]:
        # Build reliability data matrix: raters × items
        # Collect all item IDs scored by any rater
        all_items = set()
        for rd in rater_data:
            all_items.update(rd.keys())
        all_items = sorted(all_items)

        matrix = []
        for rd in rater_data:
            row = []
            for item_id in all_items:
                if item_id in rd:
                    val = rd[item_id].get(dim)
                    if dim == "hallucination" and isinstance(val, bool):
                        val = int(val)
                    row.append(val if val is not None else np.nan)
                else:
                    row.append(np.nan)
            matrix.append(row)

        matrix = np.array(matrix, dtype=float)

        # Need at least 2 non-NaN values per item for some items
        valid_cols = np.sum(~np.isnan(matrix), axis=0) >= 2
        if valid_cols.sum() < 2:
            print(f"  {dim:<16} insufficient overlap")
            continue

        level = "nominal" if dim == "hallucination" else "ordinal"
        try:
            alpha = krippendorff_alpha(matrix, level=level)
            n_items = int(valid_cols.sum())
            n_raters = len(rater_data)
            print(f"  {dim:<16} α = {alpha:.3f}  ({n_raters} raters, {n_items} items with ≥2 ratings)")
        except Exception as e:
            print(f"  {dim:<16} error: {e}")

    print(f"\n  Interpretation guide:")
    print(f"    α ≥ 0.80  reliable agreement")
    print(f"    α ≥ 0.67  tentative agreement (acceptable for some purposes)")
    print(f"    α < 0.67  disagreement / unreliable")


def analyze_human_pairwise(hr_a: dict, hr_b: dict) -> dict:
    """Pairwise analysis between two human reviewers."""
    name_a = hr_a["reviewer_id"]
    name_b = hr_b["reviewer_id"]
    ratings_a = hr_a["ratings"]
    ratings_b = hr_b["ratings"]

    overlap_ids = sorted(set(ratings_a.keys()) & set(ratings_b.keys()))
    print_section(f"{name_a} vs {name_b} ({len(overlap_ids)} overlapping items)")

    if len(overlap_ids) < 2:
        print("  Insufficient overlap for analysis.")
        return {}

    from scipy.stats import pearsonr, spearmanr

    all_kappas = {}

    print(f"\n  {'Dimension':<16} {'Kappa_w':>8} {'r':>8} {name_a + '_μ':>9} {name_b + '_μ':>9} {'Δμ':>7} {'MAE':>6}")
    print(f"  {'-'*14:<16} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*7} {'-'*6}")

    for dim in DIMENSIONS:
        a_scores = []
        b_scores = []
        for iid in overlap_ids:
            a_val = ratings_a[iid].get(dim)
            b_val = ratings_b[iid].get(dim)
            if a_val is not None and b_val is not None:
                a_scores.append(a_val)
                b_scores.append(b_val)

        if len(a_scores) < 2:
            print(f"  {dim:<16} insufficient data")
            continue

        a_arr = np.array(a_scores)
        b_arr = np.array(b_scores)

        kappa = cohens_kappa_ordinal(a_scores, b_scores)
        r_val, _ = pearsonr(a_arr, b_arr)
        a_mean = a_arr.mean()
        b_mean = b_arr.mean()
        delta = a_mean - b_mean
        mae = np.abs(a_arr - b_arr).mean()
        all_kappas[dim] = kappa

        print(f"  {dim:<16} {kappa:>8.3f} {r_val:>8.3f} {a_mean:>9.2f} {b_mean:>9.2f} {delta:>+7.2f} {mae:>6.2f}")

    # Hallucination
    a_flags = []
    b_flags = []
    for iid in overlap_ids:
        a_val = ratings_a[iid].get("hallucination")
        b_val = ratings_b[iid].get("hallucination")
        if a_val is not None and b_val is not None:
            a_flags.append(a_val)
            b_flags.append(b_val)

    if len(a_flags) >= 2:
        try:
            kappa_hall = cohens_kappa_nominal(
                [int(x) for x in a_flags],
                [int(x) for x in b_flags],
            )
        except Exception:
            kappa_hall = float("nan")
        all_kappas["hallucination"] = kappa_hall

        a_rate = sum(a_flags) / len(a_flags) * 100
        b_rate = sum(b_flags) / len(b_flags) * 100
        agreement = sum(1 for a, b in zip(a_flags, b_flags) if a == b) / len(a_flags) * 100

        print(f"\n  Hallucination:")
        print(f"    Cohen's kappa:     {kappa_hall:.3f}")
        print(f"    Raw agreement:     {agreement:.1f}%")
        print(f"    {name_a} flag rate: {sum(a_flags)}/{len(a_flags)} ({a_rate:.1f}%)")
        print(f"    {name_b} flag rate: {sum(b_flags)}/{len(b_flags)} ({b_rate:.1f}%)")

    # Agreement rates
    print(f"\n  Agreement rates:")
    print(f"  {'Dimension':<16} {'Exact':>7} {'±1':>7} {'±2':>7}")
    print(f"  {'-'*14:<16} {'-'*7} {'-'*7} {'-'*7}")
    for dim in DIMENSIONS:
        a_scores = []
        b_scores = []
        for iid in overlap_ids:
            a_val = ratings_a[iid].get(dim)
            b_val = ratings_b[iid].get(dim)
            if a_val is not None and b_val is not None:
                a_scores.append(a_val)
                b_scores.append(b_val)
        if not a_scores:
            continue
        n = len(a_scores)
        exact = sum(1 for a, b in zip(a_scores, b_scores) if a == b) / n * 100
        adj1 = sum(1 for a, b in zip(a_scores, b_scores) if abs(a - b) <= 1) / n * 100
        adj2 = sum(1 for a, b in zip(a_scores, b_scores) if abs(a - b) <= 2) / n * 100
        print(f"  {dim:<16} {exact:>6.1f}% {adj1:>6.1f}% {adj2:>6.1f}%")

    return all_kappas


def analyze_krippendorff_humans(human_reviews: list[dict]):
    """Krippendorff's alpha for human raters only (excluding Claude)."""
    print_section("Krippendorff's Alpha (humans only)")

    rater_data = [r["ratings"] for r in human_reviews]
    all_items = set()
    for rd in rater_data:
        all_items.update(rd.keys())
    all_items = sorted(all_items)

    for dim in DIMENSIONS + ["hallucination"]:
        matrix = []
        for rd in rater_data:
            row = []
            for item_id in all_items:
                if item_id in rd:
                    val = rd[item_id].get(dim)
                    if dim == "hallucination" and isinstance(val, bool):
                        val = int(val)
                    row.append(val if val is not None else np.nan)
                else:
                    row.append(np.nan)
            matrix.append(row)

        matrix = np.array(matrix, dtype=float)
        valid_cols = np.sum(~np.isnan(matrix), axis=0) >= 2
        if valid_cols.sum() < 2:
            print(f"  {dim:<16} insufficient overlap")
            continue

        level = "nominal" if dim == "hallucination" else "ordinal"
        try:
            alpha = krippendorff_alpha(matrix, level=level)
            n_items = int(valid_cols.sum())
            n_raters = len(rater_data)
            print(f"  {dim:<16} α = {alpha:.3f}  ({n_raters} raters, {n_items} items with ≥2 ratings)")
        except Exception as e:
            print(f"  {dim:<16} error: {e}")


def summary_comparison(claude: dict, human_reviews: list[dict]):
    """Summary statistics across all reviewers."""
    print_section("Summary: Mean scores by rater")

    print(f"\n  {'Rater':<16} {'n':>4} {'Acc':>6} {'Comp':>6} {'Help':>6} {'Hall%':>7}")
    print(f"  {'-'*14:<16} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

    # Claude (full 164)
    c_items = list(claude.values())
    c_n = len(c_items)
    for label, items, n in [("Claude (all)", c_items, c_n)]:
        acc = np.mean([i["accuracy"] for i in items if i.get("accuracy")])
        comp = np.mean([i["completeness"] for i in items if i.get("completeness")])
        hlp = np.mean([i["helpfulness"] for i in items if i.get("helpfulness")])
        hall = sum(1 for i in items if i.get("hallucination") is True) / n * 100
        print(f"  {label:<16} {n:>4} {acc:>6.2f} {comp:>6.2f} {hlp:>6.2f} {hall:>6.1f}%")

    for hr in human_reviews:
        reviewer = hr["reviewer_id"]
        items = list(hr["ratings"].values())
        n = len(items)
        acc = np.mean([i["accuracy"] for i in items if i.get("accuracy") is not None])
        comp = np.mean([i["completeness"] for i in items if i.get("completeness") is not None])
        hlp = np.mean([i["helpfulness"] for i in items if i.get("helpfulness") is not None])
        hall = sum(1 for i in items if i.get("hallucination") is True) / n * 100
        print(f"  {reviewer:<16} {n:>4} {acc:>6.2f} {comp:>6.2f} {hlp:>6.2f} {hall:>6.1f}%")

    # Claude on same subset as each human
    print(f"\n  {'Rater':<16} {'n':>4} {'Acc':>6} {'Comp':>6} {'Help':>6} {'Hall%':>7}  (Claude on same items)")
    print(f"  {'-'*14:<16} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for hr in human_reviews:
        reviewer = hr["reviewer_id"]
        overlap_ids = [iid for iid in hr["ratings"] if iid in claude]
        c_sub = [claude[iid] for iid in overlap_ids]
        h_sub = [hr["ratings"][iid] for iid in overlap_ids]
        n = len(overlap_ids)
        if n == 0:
            continue
        c_acc = np.mean([i["accuracy"] for i in c_sub if i.get("accuracy")])
        c_comp = np.mean([i["completeness"] for i in c_sub if i.get("completeness")])
        c_hlp = np.mean([i["helpfulness"] for i in c_sub if i.get("helpfulness")])
        c_hall = sum(1 for i in c_sub if i.get("hallucination") is True) / n * 100

        h_acc = np.mean([i["accuracy"] for i in h_sub if i.get("accuracy") is not None])
        h_comp = np.mean([i["completeness"] for i in h_sub if i.get("completeness") is not None])
        h_hlp = np.mean([i["helpfulness"] for i in h_sub if i.get("helpfulness") is not None])
        h_hall = sum(1 for i in h_sub if i.get("hallucination") is True) / n * 100

        print(f"  Claude->{reviewer:<10} {n:>4} {c_acc:>6.2f} {c_comp:>6.2f} {c_hlp:>6.2f} {c_hall:>6.1f}%")
        print(f"  {reviewer:<16} {n:>4} {h_acc:>6.2f} {h_comp:>6.2f} {h_hlp:>6.2f} {h_hall:>6.1f}%")


def main():
    print("Inter-Rater Agreement Analysis")
    print("Claude-as-Judge vs Human Expert Reviewers")
    print(f"{'='*60}")

    # Load data
    if not os.path.exists(CLAUDE_PATH):
        print(f"ERROR: Claude review not found at {CLAUDE_PATH}")
        sys.exit(1)

    claude = load_claude_reviews(CLAUDE_PATH)
    print(f"  Claude reviews loaded: {len(claude)} items")

    human_reviews = load_human_reviews(RESULTS_DIR)
    if not human_reviews:
        print(f"ERROR: No human review files (evaluation-*.json) found in {RESULTS_DIR}")
        sys.exit(1)

    for hr in human_reviews:
        print(f"  {hr['reviewer_id']} reviews loaded: {hr['sample_size']} items ({hr['file']})")

    # Summary comparison
    summary_comparison(claude, human_reviews)

    # Pairwise analysis: Claude vs each human
    all_kappas = {}
    for hr in human_reviews:
        kappas = analyze_pairwise(claude, hr)
        all_kappas[f"Claude vs {hr['reviewer_id']}"] = kappas

    # Pairwise analysis: human vs human
    if len(human_reviews) >= 2:
        for i in range(len(human_reviews)):
            for j in range(i + 1, len(human_reviews)):
                hr_a = human_reviews[i]
                hr_b = human_reviews[j]
                kappas = analyze_human_pairwise(hr_a, hr_b)
                all_kappas[f"{hr_a['reviewer_id']} vs {hr_b['reviewer_id']}"] = kappas

    # Krippendorff's alpha (all raters)
    analyze_krippendorff(claude, human_reviews)

    # Krippendorff's alpha (humans only)
    if len(human_reviews) >= 2:
        analyze_krippendorff_humans(human_reviews)

    # Overall interpretation
    print_section("Interpretation Summary")
    print(f"\n  {'Pair':<28} {'Acc':>8} {'Comp':>8} {'Help':>8} {'Hall':>8}")
    print(f"  {'-'*26:<28} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for pair, kappas in all_kappas.items():
        acc = kappas.get("accuracy", float("nan"))
        comp = kappas.get("completeness", float("nan"))
        hlp = kappas.get("helpfulness", float("nan"))
        hall = kappas.get("hallucination", float("nan"))
        print(f"  {pair:<28} {acc:>8.3f} {comp:>8.3f} {hlp:>8.3f} {hall:>8.3f}")

    print(f"\n  Landis & Koch (1977) benchmarks:")
    print(f"    < 0.00 less than chance | 0.00-0.20 slight | 0.21-0.40 fair")
    print(f"    0.41-0.60 moderate | 0.61-0.80 substantial | 0.81+ almost perfect")


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) interpretation."""
    if k < 0:
        return "less than chance"
    elif k < 0.21:
        return "slight"
    elif k < 0.41:
        return "fair"
    elif k < 0.61:
        return "moderate"
    elif k < 0.81:
        return "substantial"
    else:
        return "almost perfect"


if __name__ == "__main__":
    main()
