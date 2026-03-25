# Reproducibility Pack: A Cost-Effective Architecture for Enterprise LLM Applications

**Paper**: *A Cost-Effective Architecture for Enterprise LLM Applications: Balancing Competing Requirements through RAG-Augmented CPU-Only Inference*

**Authors**: Andrew Le Gear, Peter Hall, Brian Collins, Muslim Chochlov

**Venue**: 52nd Euromicro Conference on Software Engineering and Advanced Applications (SEAA 2026), DAIDE Track, Krakow, Poland

**Live System**: [https://upstream.exchange](https://upstream.exchange)

---

## Repository Structure

```
reproducibility-pack/
├── README.md                          # This file
├── evaluation-data/
│   ├── latency_benchmark.csv          # Raw end-to-end response times (N=50)
│   ├── quality_evaluation_summary.csv # Aggregated quality metrics across 3 methods
│   ├── cost_comparison.csv            # Monthly cost data for deployment options
│   └── cache_performance.csv          # Response times by query type (hit/miss)
├── knowledge-base/
│   └── README.md                      # Description of the 164 Q&A corpus structure
└── scripts/
    └── README.md                      # Placeholder for analysis scripts
```

## Overview

This pack contains the evaluation data supporting the three-dimensional assessment of the CPU-only LLM reference architecture described in the paper:

1. **Latency benchmark** — End-to-end response times for 50 paraphrased queries against a live deployment
2. **Output quality evaluation** — Multi-method quality assessment (automated metrics, LLM-as-judge, human expert review) across 164 Q&A pairs
3. **Financial cost analysis** — Azure pricing comparison of CPU vs. GPU deployment options

## Prerequisites

To reproduce the evaluation:

- **Python 3.10+** with the following packages:
  - `sentence-transformers` (for embedding generation)
  - `faiss-cpu` (for vector similarity search)
  - `llama-cpp-python` (for CPU-based LLM inference)
  - `rouge-score`, `bert-score` (for automated quality metrics)
- **Model file**: `gemma-2b-it.Q4_K_M.gguf` (4-bit quantized Gemma 2B, available from HuggingFace)
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (downloaded automatically)
- A machine with at least 8 vCPUs and 32 GB RAM (or an Azure P3v3 App Service Plan)

## Reproduction Steps

### 1. Latency Benchmark

The latency benchmark was conducted on February 8, 2026 against the live production deployment.

1. Prepare 50 paraphrased questions derived from the knowledge base (semantic variants, not keyword matches) plus 10 out-of-scope questions.
2. Submit each query to the system endpoint and record the end-to-end response time.
3. Raw response times are provided in `evaluation-data/latency_benchmark.csv`.
4. Summary statistics (mean, median, P95, IQR) can be computed from the raw data.

**Key result**: Mean response time of 21.0s, P95 of 29.0s, 100% success rate.

### 2. Output Quality Evaluation

The quality evaluation used three complementary methods:

**a) Automated Metrics (N=164)**
1. Generate responses for all 164 knowledge base items.
2. Compute ROUGE-1/2/L and BERTScore F₁ against ground-truth answers.
3. Results are summarised in `evaluation-data/quality_evaluation_summary.csv`.

**b) LLM-as-Judge (N=164)**
1. Submit each response + ground-truth pair to Claude Sonnet 4 with a structured evaluation prompt.
2. Collect accuracy ratings (1–5) and hallucination flags.
3. Aggregate results are in the summary CSV.

**c) Human Expert Review (N=50 per reviewer, 15 overlap)**
1. Two domain experts independently rated subsets of 50 items each on accuracy (1–5) and flagged hallucinations.
2. 15 items overlapped for inter-rater agreement (Cohen's κ).
3. Aggregate results are in the summary CSV.

**Key results**: Human accuracy 4.49–4.76/5, hallucination rates 2–6%, Cohen's κ = 0.531 (accuracy), 0.571 (hallucination).

### 3. Financial Cost Analysis

1. Cost data was sourced from Microsoft Azure official pricing pages (accessed January 29, 2026).
2. All prices reflect 3-year commitment pricing for the Central US region.
3. Comparison data is in `evaluation-data/cost_comparison.csv`.

**Key result**: CPU-only deployment (P3v3) at $203.67/month vs. V100 GPU VM at $1,188.61/month (5.8× saving).

## Data File Descriptions

| File | Contents | Format |
|------|----------|--------|
| `latency_benchmark.csv` | Per-query response times, query type (in-scope/out-of-scope), HTTP status | CSV, 60 rows |
| `quality_evaluation_summary.csv` | Aggregated quality metrics across all 3 evaluation methods | CSV, summary |
| `cost_comparison.csv` | Deployment option specifications and monthly costs | CSV, 3 rows |
| `cache_performance.csv` | Mean/median response times split by cache hit vs. miss | CSV, 2 rows |

## Limitations

- The latency benchmark reflects a specific point-in-time measurement (February 8, 2026) on the production system. Results may vary under different load conditions.
- Human evaluation scores are inherently subjective. The inter-rater agreement statistics (Cohen's κ ≈ 0.53–0.57) indicate moderate agreement.
- Cost data reflects Azure pricing at the time of access and may change.
- The knowledge base content (164 Q&A pairs) is proprietary to the Upstream platform and cannot be redistributed. The structure and categories are described in `knowledge-base/README.md`.

## Citation

If you use this data in your research, please cite the accompanying paper.

## License

This reproducibility pack is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
