# Reproducibility Pack: A Cost-Effective Architecture for Enterprise LLM Applications

**Paper**: *A Cost-Effective Architecture for Enterprise LLM Applications: Balancing Competing Requirements through RAG-Augmented CPU-Only Inference*

**Authors**: Andrew Le Gear, Peter Hall, Brian Collins, Muslim Chochlov

**Venue**: 52nd Euromicro Conference on Software Engineering and Advanced Applications (SEAA 2026), DAIDE Track, Krakow, Poland

**Live System**: [https://upstream.exchange](https://upstream.exchange)

---

## Repository Structure

```
reproducibility-pack/
├── README.md                                          # This file
├── EVALUATION_REPORT.md                               # Full companion analysis document
├── evaluation-data/
│   ├── dataset.json                                   # Complete 164-item dataset (questions,
│   │                                                  #   ground truth, chatbot responses, latency)
│   ├── automated_metrics.json                         # Per-item ROUGE/BERTScore (N=164)
│   ├── faithfulness_scores.json                       # Per-item Claude faithfulness scores (N=164)
│   ├── claude_review_scores.json                      # Per-item Claude accuracy/completeness/
│   │                                                  #   helpfulness/hallucination scores (N=164)
│   ├── evaluation-Andy-2026-03-23.json                # Human Reviewer A per-item scores (N=50)
│   ├── evaluation-TimLovett-2026-03-24.json           # Human Reviewer B per-item scores (N=50)
│   ├── latency_benchmark.csv                          # Raw end-to-end response times (N=50)
│   ├── quality_evaluation_summary.csv                 # Aggregated quality metrics
│   ├── cost_comparison.csv                            # Monthly cost data for deployment options
│   └── cache_performance.csv                          # Response times by query type (hit/miss)
├── expert-review-dashboard/
│   ├── index.html                                     # Expert review web app (open in browser)
│   ├── app.js                                         # Dashboard application logic
│   └── style.css                                      # Dashboard styling
├── scripts/
│   ├── build_dataset.py                               # Script to rebuild dataset from live endpoint
│   ├── analyze_hallucinations.py                      # Hallucination categorisation analysis
│   ├── requirements.txt                               # Python dependencies for data collection
│   └── automated-metrics/
│       ├── compute_metrics.py                         # ROUGE/BERTScore computation
│       ├── compute_faithfulness.py                    # Claude faithfulness scoring
│       ├── compute_claude_review.py                   # Claude multi-dimensional review
│       ├── compute_agreement.py                       # Inter-rater agreement analysis
│       ├── deep_analysis.py                           # Correlation and pattern analysis
│       └── requirements.txt                           # Python dependencies
├── knowledge-base/
│   └── README.md                                      # Description of the 164 Q&A corpus
└── (see also: sources/implementation/ at repo root)   # Reference implementation source code
```

## Overview

This pack contains the complete evaluation data, analysis scripts, expert review dashboard, and companion analysis document supporting the paper. The evaluation covers three dimensions:

1. **Latency benchmark** — End-to-end response times for 50 paraphrased queries
2. **Output quality evaluation** — Multi-method assessment (automated metrics, LLM-as-judge, human expert review) across 164 Q&A pairs
3. **Financial cost analysis** — Azure pricing comparison of CPU vs. GPU deployment options

## Evaluation Data

### Full Dataset (`evaluation-data/dataset.json`)

The complete evaluation dataset containing all 164 FAQ items with:
- Original question from the knowledge base
- Ground-truth answer from the knowledge base
- Chatbot-generated response from the live system
- End-to-end response time in seconds

This dataset was built by querying the live production endpoint on March 23, 2026 using `scripts/build_dataset.py`.

### Human Expert Review Data (`evaluation-data/evaluation-*.json`)

Two domain experts independently evaluated random subsets of 50 items each:

| File | Reviewer | Items Rated | Date |
|------|----------|-------------|------|
| `evaluation-Andy-2026-03-23.json` | Reviewer A  | 50 | 2026-03-23 |
| `evaluation-TimLovett-2026-03-24.json` | Reviewer B | 50 | 2026-03-24 |

Each file contains per-item ratings on:
- **Accuracy** (1–5 Likert scale)
- **Completeness** (1–5 Likert scale)
- **Helpfulness** (1–5 Likert scale)
- **Hallucination** (boolean flag)
- **Comment** (optional free-text)

15 items overlap between reviewers for inter-rater agreement analysis. Items were presented in randomised order.

### LLM-as-Judge Results (`evaluation-data/`)

| File | Description | Items |
|------|-------------|-------|
| `automated_metrics.json` | Per-item ROUGE-1/2/L and BERTScore F₁ | 164 |
| `faithfulness_scores.json` | Per-item faithfulness score (1–5), unsupported claims list, reasoning | 164 |
| `claude_review_scores.json` | Per-item accuracy/completeness/helpfulness (1–5), hallucination flag, reasoning | 164 |

All LLM-as-judge evaluations used Claude Sonnet 4 (`claude-sonnet-4-20250514`).

### Expert Review Dashboard (`expert-review-dashboard/`)

A self-contained web application used to collect human expert ratings:

- Open `index.html` in a browser (no server required \u2014 pure client-side)
- Loads the dataset from `../evaluation-data/dataset.json`
- Presents items in randomised order with ground truth side-by-side
- Supports configurable sample sizes (30, 50, or all 164)
- Saves progress to browser localStorage; exports results as JSON
- Follows Van der Lee et al. (2021) best practices for human evaluation of generated text

### Companion Analysis Document (`EVALUATION_REPORT.md`)

A comprehensive companion report (~17 sections) providing:

- Full methodology for all four evaluation methods
- Complete per-method results with distribution analyses
- Inter-rater agreement analysis (Cohen's κ, Krippendorff's α)
- Deep correlation analysis (automated metrics vs. human scores, latency vs. quality, response length effects)
- Worst/best item analysis
- Hallucination categorisation (fabrication, overgeneralisation, refusal/deflection)
- Limitations and threats to validity
- Reproducibility guide

## Prerequisites

To reproduce the evaluation:

- **Python 3.10+** with packages listed in `scripts/automated-metrics/requirements.txt`:
  - `rouge-score`, `bert-score` (automated quality metrics)
  - `scikit-learn`, `krippendorff`, `scipy`, `numpy` (agreement analysis)
  - `anthropic` (LLM-as-judge evaluations — requires `ANTHROPIC_API_KEY`)
- For dataset rebuilding: `requests` (listed in `scripts/requirements.txt`)
- **Model file**: `gemma-2b-it.Q4_K_M.gguf` (4-bit quantized Gemma 2B)
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (auto-downloaded)
- A machine with at least 8 vCPUs and 32 GB RAM (or Azure P3v3 App Service Plan)

## Reproduction Steps

### 1. Latency Benchmark

The latency benchmark was conducted on February 8, 2026 against the live production deployment.

1. Prepare 50 paraphrased questions (semantic variants, not keyword matches) plus 10 out-of-scope questions.
2. Submit each query to the system endpoint and record the end-to-end response time.
3. Raw response times are in `reproducibility-pack/evaluation-data/latency_benchmark.csv`.

**Key result**: Mean 21.0s, P95 29.0s, 100% success rate.

### 2. Dataset Construction

```bash
cd scripts
pip install -r requirements.txt
python build_dataset.py --endpoint <YOUR_ENDPOINT>
```

This queries the live chatbot for all 164 knowledge base items and saves `dataset.json`. The script has resume capability — it saves after each item and skips already-answered questions if interrupted.

### 3. Automated Quality Metrics

```bash
cd scripts/automated-metrics
pip install -r requirements.txt
python compute_metrics.py           # ROUGE + BERTScore
python compute_faithfulness.py      # Claude faithfulness (requires ANTHROPIC_API_KEY)
python compute_claude_review.py     # Claude multi-dimensional review
```

All scripts have resume capability and save results incrementally to `evaluation-data/`.

### 4. Human Expert Review

1. Open `expert-review-dashboard/index.html` in a web browser.
2. Enter a reviewer name and select sample size (50 recommended).
3. Rate each item on accuracy, completeness, helpfulness (1–5) and flag hallucinations.
4. Click "Finish & Export" to download the results JSON.
5. Place the exported file in `evaluation-data/`.

### 5. Inter-Rater Agreement & Deep Analysis

```bash
python compute_agreement.py         # Cohen's κ, Krippendorff's α
python deep_analysis.py             # Correlation and pattern analysis
```

### 6. Financial Cost Analysis

Cost data was sourced from Microsoft Azure pricing pages (accessed January 29, 2026). All prices reflect 3-year commitment pricing for Central US region. Data is in `reproducibility-pack/evaluation-data/cost_comparison.csv`.

**Key result**: CPU-only (P3v3) $203.67/month vs. V100 GPU $1,188.61/month (5.8× saving).

## Summary of Key Results

| Dimension | Key Metric | Value |
|-----------|-----------|-------|
| Latency | Mean response time | 21.0s |
| Latency | P95 response time | 29.0s |
| Quality | Human accuracy (Reviewer A) | 4.76 / 5 |
| Quality | Human accuracy (Reviewer B) | 4.49 / 5 |
| Quality | Human hallucination rate | 2–6% |
| Quality | BERTScore F₁ | 0.81 |
| Quality | Inter-rater agreement (accuracy) | κ = 0.531 |
| Quality | Inter-rater agreement (hallucination) | κ = 0.571 |
| Cost | CPU monthly cost (P3v3) | $203.67 |
| Cost | V100 GPU monthly cost | $1,188.61 |

## Limitations

- Latency benchmark reflects a point-in-time measurement (Feb 8, 2026). Results may vary under different load.
- Human evaluation scores are subjective. Inter-rater agreement (κ ≈ 0.53–0.57) indicates moderate agreement.
- Cost data reflects Azure pricing at time of access and may change.
- The API key in dataset metadata has been redacted (`<REDACTED>`). To rebuild the dataset, supply your own endpoint.

## Citation

If you use this data in your research, please cite the accompanying paper.

## License

This reproducibility pack is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
