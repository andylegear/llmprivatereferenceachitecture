"""
Deep analysis of evaluation data: correlations, patterns, and insights.

Analyses:
1. Automated metrics (ROUGE/BERTScore) vs human scores — validation
2. Latency vs quality — does inference time predict output quality?
3. Response length vs quality — do longer responses score higher?
4. Ground-truth length analysis — does source complexity affect quality?
5. Cross-measure hallucination consistency — faithfulness vs Claude vs human
6. Worst/best item analysis — which items are universally good or bad?

All data is loaded from existing results files. No API calls needed.
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "evaluation-data")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "evaluation-data")

DATASET_PATH = os.path.join(DATA_DIR, "dataset.json")
METRICS_PATH = os.path.join(RESULTS_DIR, "automated_metrics.json")
FAITHFULNESS_PATH = os.path.join(RESULTS_DIR, "faithfulness_scores.json")
CLAUDE_REVIEW_PATH = os.path.join(RESULTS_DIR, "claude_review_scores.json")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_by_id(items, key="id"):
    return {item[key]: item for item in items}


def print_section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def safe_nanmean(values):
    """Compute mean ignoring None and NaN."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return np.mean(clean) if clean else np.nan


def corr_report(x, y, label_x, label_y, n_label="n"):
    """Print Pearson and Spearman correlation between two arrays."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        print(f"    {label_x} vs {label_y}: insufficient data ({len(x)} points)")
        return
    r, p_r = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)
    sig_r = "*" if p_r < 0.05 else ""
    sig_rho = "*" if p_rho < 0.05 else ""
    print(f"    {label_x:.<30s} vs {label_y:.<15s}  r={r:+.3f}{sig_r:1s}  ρ={rho:+.3f}{sig_rho:1s}  ({n_label}={len(x)})")


def main():
    # Load all data
    dataset = load_json(DATASET_PATH)
    ds_items = index_by_id(dataset["items"])

    metrics = load_json(METRICS_PATH)
    met_items = index_by_id(metrics["items"])

    faith = load_json(FAITHFULNESS_PATH)
    faith_items = index_by_id(faith["items"])

    claude_rev = load_json(CLAUDE_REVIEW_PATH)
    claude_items = index_by_id(claude_rev["items"])

    # Load human reviews
    human_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "evaluation-*.json")))
    human_reviews = []
    for fpath in human_files:
        data = load_json(fpath)
        human_reviews.append({
            "reviewer_id": data.get("reviewer_id", os.path.basename(fpath)),
            "ratings": {r["item_id"]: r for r in data["ratings"]},
        })

    # Merge all data by item ID
    all_ids = sorted(ds_items.keys())

    print("Deep Analysis of Evaluation Data")
    print(f"{'='*65}")
    print(f"  Dataset: {len(all_ids)} items")
    print(f"  Human reviewers: {len(human_reviews)} ({', '.join(h['reviewer_id'] for h in human_reviews)})")

    # =========================================================================
    # 1. AUTOMATED METRICS vs HUMAN SCORES
    # =========================================================================
    print_section("1. Automated Metrics vs Human Scores")
    print("  Do ROUGE/BERTScore predict human satisfaction?\n")

    metric_names = ["rouge1_f1", "rouge2_f1", "rougeL_f1",
                    "bertscore_precision", "bertscore_recall", "bertscore_f1"]
    human_dims = ["accuracy", "completeness", "helpfulness"]

    # Aggregate human scores (average across reviewers for overlapping items)
    human_agg = {}
    for iid in all_ids:
        scores = {d: [] for d in human_dims}
        for hr in human_reviews:
            if iid in hr["ratings"]:
                for d in human_dims:
                    v = hr["ratings"][iid].get(d)
                    if v is not None:
                        scores[d].append(v)
        if any(scores[d] for d in human_dims):
            human_agg[iid] = {d: np.mean(scores[d]) if scores[d] else np.nan for d in human_dims}

    # Also include Claude scores for full coverage
    print("  a) Automated metrics vs Claude review (n=164):")
    for metric in metric_names:
        for dim in human_dims:
            x_vals, y_vals = [], []
            for iid in all_ids:
                if iid in met_items and iid in claude_items:
                    m = met_items[iid].get(metric)
                    c = claude_items[iid].get(dim)
                    if m is not None and c is not None:
                        x_vals.append(m)
                        y_vals.append(c)
            corr_report(x_vals, y_vals, metric, f"claude_{dim}", "n")

    print(f"\n  b) Automated metrics vs Human scores (n={len(human_agg)}):")
    for metric in metric_names:
        for dim in human_dims:
            x_vals, y_vals = [], []
            for iid, hscores in human_agg.items():
                if iid in met_items and not np.isnan(hscores[dim]):
                    m = met_items[iid].get(metric)
                    if m is not None:
                        x_vals.append(m)
                        y_vals.append(hscores[dim])
            corr_report(x_vals, y_vals, metric, f"human_{dim}", "n")

    # =========================================================================
    # 2. LATENCY vs QUALITY
    # =========================================================================
    print_section("2. Latency vs Quality")
    print("  Does inference time predict output quality?\n")

    latencies = {iid: ds_items[iid]["elapsed_time_seconds"] for iid in all_ids}

    # vs Claude scores
    print("  a) Latency vs Claude review:")
    for dim in human_dims + ["faithfulness"]:
        x_vals, y_vals = [], []
        for iid in all_ids:
            lat = latencies[iid]
            if dim == "faithfulness":
                score = faith_items.get(iid, {}).get("faithfulness_score")
            else:
                score = claude_items.get(iid, {}).get(dim)
            if score is not None:
                x_vals.append(lat)
                y_vals.append(score)
        corr_report(x_vals, y_vals, "latency_s", dim, "n")

    # vs automated metrics
    print(f"\n  b) Latency vs automated metrics:")
    for metric in ["rouge1_f1", "bertscore_f1"]:
        x_vals, y_vals = [], []
        for iid in all_ids:
            lat = latencies[iid]
            m = met_items.get(iid, {}).get(metric)
            if m is not None:
                x_vals.append(lat)
                y_vals.append(m)
        corr_report(x_vals, y_vals, "latency_s", metric, "n")

    # Latency quartile breakdown
    lat_vals = np.array([latencies[iid] for iid in all_ids])
    q25, q50, q75 = np.percentile(lat_vals, [25, 50, 75])
    print(f"\n  c) Quality by latency quartile:")
    print(f"     Quartile boundaries: Q1<{q25:.1f}s, Q2<{q50:.1f}s, Q3<{q75:.1f}s, Q4>{q75:.1f}s")
    quartile_labels = ["Q1 (fast)", "Q2", "Q3", "Q4 (slow)"]
    quartile_bounds = [(-1, q25), (q25, q50), (q50, q75), (q75, 999)]
    print(f"     {'Quartile':<12} {'n':>4} {'Acc':>6} {'Comp':>6} {'Help':>6} {'Faith':>6} {'ROUGE1':>7} {'BERTSc':>7}")
    for label, (lo, hi) in zip(quartile_labels, quartile_bounds):
        ids_in_q = [iid for iid in all_ids if lo < latencies[iid] <= hi]
        n = len(ids_in_q)
        acc = safe_nanmean([claude_items.get(i, {}).get("accuracy", np.nan) for i in ids_in_q])
        comp = safe_nanmean([claude_items.get(i, {}).get("completeness", np.nan) for i in ids_in_q])
        hlp = safe_nanmean([claude_items.get(i, {}).get("helpfulness", np.nan) for i in ids_in_q])
        fth = safe_nanmean([faith_items.get(i, {}).get("faithfulness_score", np.nan) for i in ids_in_q])
        r1 = safe_nanmean([met_items.get(i, {}).get("rouge1_f1", np.nan) for i in ids_in_q])
        bs = safe_nanmean([met_items.get(i, {}).get("bertscore_f1", np.nan) for i in ids_in_q])
        print(f"     {label:<12} {n:>4} {acc:>6.2f} {comp:>6.2f} {hlp:>6.2f} {fth:>6.2f} {r1:>7.3f} {bs:>7.3f}")

    # =========================================================================
    # 3. RESPONSE LENGTH vs QUALITY
    # =========================================================================
    print_section("3. Response Length vs Quality")
    print("  Do longer chatbot responses score higher?\n")

    resp_lengths = {iid: len(ds_items[iid]["chatbot_response"].split()) for iid in all_ids}
    gt_lengths = {iid: len(ds_items[iid]["ground_truth"].split()) for iid in all_ids}
    length_ratios = {iid: resp_lengths[iid] / max(gt_lengths[iid], 1) for iid in all_ids}

    print("  a) Response word count vs scores:")
    for dim in human_dims:
        x, y = [], []
        for iid in all_ids:
            score = claude_items.get(iid, {}).get(dim)
            if score is not None:
                x.append(resp_lengths[iid])
                y.append(score)
        corr_report(x, y, "response_words", f"claude_{dim}", "n")

    print(f"\n  b) Response/Ground-truth length ratio vs scores:")
    for dim in human_dims:
        x, y = [], []
        for iid in all_ids:
            score = claude_items.get(iid, {}).get(dim)
            if score is not None:
                x.append(length_ratios[iid])
                y.append(score)
        corr_report(x, y, "length_ratio", f"claude_{dim}", "n")

    for metric in ["rouge1_f1", "bertscore_f1"]:
        x, y = [], []
        for iid in all_ids:
            m = met_items.get(iid, {}).get(metric)
            if m is not None:
                x.append(resp_lengths[iid])
                y.append(m)
        corr_report(x, y, "response_words", metric, "n")

    # Length stats by score bucket
    print(f"\n  c) Mean response length by Claude accuracy score:")
    for score in range(1, 6):
        ids = [iid for iid in all_ids if claude_items.get(iid, {}).get("accuracy") == score]
        if ids:
            mean_len = np.mean([resp_lengths[iid] for iid in ids])
            mean_gt = np.mean([gt_lengths[iid] for iid in ids])
            mean_ratio = np.mean([length_ratios[iid] for iid in ids])
            print(f"     Score {score}: n={len(ids):>3}, resp={mean_len:>5.0f} words, GT={mean_gt:>5.0f} words, ratio={mean_ratio:.2f}")

    # =========================================================================
    # 4. GROUND-TRUTH COMPLEXITY
    # =========================================================================
    print_section("4. Ground-Truth Complexity vs Quality")
    print("  Do longer/more complex source answers lead to worse scores?\n")

    print("  a) Ground-truth word count vs scores:")
    for dim in human_dims:
        x, y = [], []
        for iid in all_ids:
            score = claude_items.get(iid, {}).get(dim)
            if score is not None:
                x.append(gt_lengths[iid])
                y.append(score)
        corr_report(x, y, "gt_words", f"claude_{dim}", "n")

    for metric in ["rouge1_f1", "bertscore_f1"]:
        x, y = [], []
        for iid in all_ids:
            m = met_items.get(iid, {}).get(metric)
            if m is not None:
                x.append(gt_lengths[iid])
                y.append(m)
        corr_report(x, y, "gt_words", metric, "n")

    # GT length quartile breakdown
    gt_vals = np.array([gt_lengths[iid] for iid in all_ids])
    g25, g50, g75 = np.percentile(gt_vals, [25, 50, 75])
    print(f"\n  b) Quality by ground-truth length quartile:")
    print(f"     Quartile boundaries: Q1<{g25:.0f}w, Q2<{g50:.0f}w, Q3<{g75:.0f}w, Q4>{g75:.0f}w")
    quartile_bounds = [(-1, g25), (g25, g50), (g50, g75), (g75, 9999)]
    quartile_labels = ["Q1 (short)", "Q2", "Q3", "Q4 (long)"]
    print(f"     {'Quartile':<12} {'n':>4} {'Acc':>6} {'Comp':>6} {'Help':>6} {'ROUGE1':>7} {'BERTSc':>7} {'Hall%':>6}")
    for label, (lo, hi) in zip(quartile_labels, quartile_bounds):
        ids_in_q = [iid for iid in all_ids if lo < gt_lengths[iid] <= hi]
        n = len(ids_in_q)
        acc = safe_nanmean([claude_items.get(i, {}).get("accuracy", np.nan) for i in ids_in_q])
        comp = safe_nanmean([claude_items.get(i, {}).get("completeness", np.nan) for i in ids_in_q])
        hlp = safe_nanmean([claude_items.get(i, {}).get("helpfulness", np.nan) for i in ids_in_q])
        r1 = safe_nanmean([met_items.get(i, {}).get("rouge1_f1", np.nan) for i in ids_in_q])
        bs = safe_nanmean([met_items.get(i, {}).get("bertscore_f1", np.nan) for i in ids_in_q])
        hall = sum(1 for i in ids_in_q if claude_items.get(i, {}).get("hallucination") is True)
        hall_pct = hall / n * 100 if n else 0
        print(f"     {label:<12} {n:>4} {acc:>6.2f} {comp:>6.2f} {hlp:>6.2f} {r1:>7.3f} {bs:>7.3f} {hall_pct:>5.1f}%")

    # =========================================================================
    # 5. CROSS-MEASURE HALLUCINATION CONSISTENCY
    # =========================================================================
    print_section("5. Cross-Measure Hallucination Consistency")
    print("  How do faithfulness score, Claude binary flag, and human flags relate?\n")

    # Faithfulness score vs Claude binary flag
    faith_when_flag = []
    faith_when_no_flag = []
    for iid in all_ids:
        fs = faith_items.get(iid, {}).get("faithfulness_score")
        cf = claude_items.get(iid, {}).get("hallucination")
        if fs is not None and cf is not None:
            if cf:
                faith_when_flag.append(fs)
            else:
                faith_when_no_flag.append(fs)

    print(f"  a) Faithfulness score when Claude flags hallucination:")
    print(f"     Flagged:     n={len(faith_when_flag):>3}, mean={np.mean(faith_when_flag):.2f}, median={np.median(faith_when_flag):.0f}")
    print(f"     Not flagged: n={len(faith_when_no_flag):>3}, mean={np.mean(faith_when_no_flag):.2f}, median={np.median(faith_when_no_flag):.0f}")

    # Faithfulness score distribution by Claude flag
    print(f"\n     Faithfulness distribution by Claude flag:")
    print(f"     {'Score':>5} {'Flagged':>8} {'Not flagged':>12}")
    for s in range(1, 6):
        f_count = faith_when_flag.count(s)
        nf_count = faith_when_no_flag.count(s)
        print(f"     {s:>5} {f_count:>8} {nf_count:>12}")

    # Human hallucination flags vs faithfulness
    print(f"\n  b) Faithfulness score by human hallucination flag:")
    human_hall_true = []
    human_hall_false = []
    for hr in human_reviews:
        for iid, rating in hr["ratings"].items():
            fs = faith_items.get(iid, {}).get("faithfulness_score")
            hf = rating.get("hallucination")
            if fs is not None and hf is not None:
                if hf:
                    human_hall_true.append(fs)
                else:
                    human_hall_false.append(fs)
    if human_hall_true:
        print(f"     Human=True:  n={len(human_hall_true):>3}, mean={np.mean(human_hall_true):.2f}")
    else:
        print(f"     Human=True:  n=  0 (too few human hallucination flags)")
    print(f"     Human=False: n={len(human_hall_false):>3}, mean={np.mean(human_hall_false):.2f}")

    # Three-way agreement on overlapping items
    print(f"\n  c) Three-way hallucination agreement (items scored by all methods):")
    combos = {"all_agree_no": 0, "all_agree_yes": 0, "claude_only": 0,
              "faith_only": 0, "human_only": 0, "claude_faith_not_human": 0, "other": 0}
    three_way_n = 0
    for hr in human_reviews:
        for iid, rating in hr["ratings"].items():
            cf = claude_items.get(iid, {}).get("hallucination")
            fs = faith_items.get(iid, {}).get("faithfulness_score")
            hf = rating.get("hallucination")
            if cf is not None and fs is not None and hf is not None:
                three_way_n += 1
                faith_flag = fs <= 3  # faithfulness <=3 = likely hallucination
                if not cf and not faith_flag and not hf:
                    combos["all_agree_no"] += 1
                elif cf and faith_flag and hf:
                    combos["all_agree_yes"] += 1
                elif cf and not hf and not faith_flag:
                    combos["claude_only"] += 1
                elif cf and faith_flag and not hf:
                    combos["claude_faith_not_human"] += 1
                elif not cf and not hf and faith_flag:
                    combos["faith_only"] += 1
                elif hf and not cf and not faith_flag:
                    combos["human_only"] += 1
                else:
                    combos["other"] += 1
    print(f"     Total items with all 3 measures: {three_way_n}")
    for k, v in combos.items():
        pct = v / three_way_n * 100 if three_way_n else 0
        print(f"     {k:<30s} {v:>4} ({pct:>5.1f}%)")

    # =========================================================================
    # 6. BEST AND WORST ITEMS
    # =========================================================================
    print_section("6. Best and Worst Items")

    # Composite score (average of Claude accuracy, completeness, helpfulness)
    composite = {}
    for iid in all_ids:
        ci = claude_items.get(iid, {})
        vals = [ci.get(d) for d in human_dims if ci.get(d) is not None]
        if vals:
            composite[iid] = np.mean(vals)

    sorted_items = sorted(composite.items(), key=lambda x: x[1])

    print(f"\n  a) 10 WORST items (lowest Claude composite):")
    print(f"     {'ID':>4} {'Comp':>5} {'Acc':>4} {'Cmp':>4} {'Hlp':>4} {'Hall':>5} {'Faith':>5} {'R1':>5} {'Lat':>5} Question")
    for iid, comp_score in sorted_items[:10]:
        ci = claude_items.get(iid, {})
        fi = faith_items.get(iid, {})
        di = ds_items.get(iid, {})
        hall = "Y" if ci.get("hallucination") else "N"
        faith_s = fi.get("faithfulness_score", "-")
        r1 = met_items.get(iid, {}).get("rouge1_f1", 0)
        lat = di.get("elapsed_time_seconds", 0)
        q = di.get("question", "?")[:50]
        print(f"     {iid:>4} {comp_score:>5.2f} {ci.get('accuracy',''):>4} {ci.get('completeness',''):>4} {ci.get('helpfulness',''):>4} {hall:>5} {faith_s:>5} {r1:>5.2f} {lat:>5.1f} {q}")

    print(f"\n  b) 10 BEST items (highest Claude composite):")
    print(f"     {'ID':>4} {'Comp':>5} {'Acc':>4} {'Cmp':>4} {'Hlp':>4} {'Hall':>5} {'Faith':>5} {'R1':>5} {'Lat':>5} Question")
    for iid, comp_score in sorted_items[-10:]:
        ci = claude_items.get(iid, {})
        fi = faith_items.get(iid, {})
        di = ds_items.get(iid, {})
        hall = "Y" if ci.get("hallucination") else "N"
        faith_s = fi.get("faithfulness_score", "-")
        r1 = met_items.get(iid, {}).get("rouge1_f1", 0)
        lat = di.get("elapsed_time_seconds", 0)
        q = di.get("question", "?")[:50]
        print(f"     {iid:>4} {comp_score:>5.2f} {ci.get('accuracy',''):>4} {ci.get('completeness',''):>4} {ci.get('helpfulness',''):>4} {hall:>5} {faith_s:>5} {r1:>5.2f} {lat:>5.1f} {q}")

    # Items where humans and Claude disagree most
    print(f"\n  c) Largest human-Claude disagreements (by accuracy):")
    print(f"     {'ID':>4} {'H_Acc':>6} {'C_Acc':>6} {'Δ':>4} Question")
    disagreements = []
    for hr in human_reviews:
        for iid, rating in hr["ratings"].items():
            h_acc = rating.get("accuracy")
            c_acc = claude_items.get(iid, {}).get("accuracy")
            if h_acc is not None and c_acc is not None:
                disagreements.append((iid, h_acc, c_acc, h_acc - c_acc, ds_items[iid]["question"]))
    disagreements.sort(key=lambda x: abs(x[3]), reverse=True)
    seen = set()
    count = 0
    for iid, h, c, delta, q in disagreements:
        if iid not in seen and count < 10:
            print(f"     {iid:>4} {h:>6} {c:>6} {delta:>+4} {q[:55]}")
            seen.add(iid)
            count += 1

    # =========================================================================
    # 7. METRIC INTERCORRELATION MATRIX
    # =========================================================================
    print_section("7. Metric Intercorrelation Matrix")
    print("  How do all measures relate to each other?\n")

    all_measures = {}
    for iid in all_ids:
        all_measures[iid] = {
            "rouge1": met_items.get(iid, {}).get("rouge1_f1"),
            "rouge2": met_items.get(iid, {}).get("rouge2_f1"),
            "rougeL": met_items.get(iid, {}).get("rougeL_f1"),
            "bertF1": met_items.get(iid, {}).get("bertscore_f1"),
            "c_acc": claude_items.get(iid, {}).get("accuracy"),
            "c_comp": claude_items.get(iid, {}).get("completeness"),
            "c_help": claude_items.get(iid, {}).get("helpfulness"),
            "faith": faith_items.get(iid, {}).get("faithfulness_score"),
            "latency": latencies.get(iid),
            "resp_len": resp_lengths.get(iid),
            "gt_len": gt_lengths.get(iid),
        }

    measure_names = ["rouge1", "rouge2", "rougeL", "bertF1", "c_acc", "c_comp", "c_help", "faith", "latency", "resp_len", "gt_len"]
    short_names = ["R1", "R2", "RL", "BF1", "Acc", "Cmp", "Hlp", "Fth", "Lat", "RLn", "GLn"]

    # Print header
    print(f"  {'':>6}", end="")
    for sn in short_names:
        print(f" {sn:>5}", end="")
    print()

    for i, m1 in enumerate(measure_names):
        print(f"  {short_names[i]:>6}", end="")
        for j, m2 in enumerate(measure_names):
            if j <= i:
                x = [all_measures[iid][m1] for iid in all_ids if all_measures[iid][m1] is not None and all_measures[iid][m2] is not None]
                y = [all_measures[iid][m2] for iid in all_ids if all_measures[iid][m1] is not None and all_measures[iid][m2] is not None]
                if len(x) >= 3:
                    r, p = spearmanr(x, y)
                    marker = "*" if p < 0.05 else " "
                    print(f" {r:>4.2f}{marker}", end="")
                else:
                    print(f"   -- ", end="")
            else:
                print(f"      ", end="")
        print()

    print(f"\n  * = p < 0.05 (Spearman ρ)")


if __name__ == "__main__":
    main()
