# Quality Evaluation of a CPU-Only LLM FAQ Chatbot: Methods, Results, and Analysis

**System Under Test**: Upstream Exchange FAQ Chatbot (Gemma 2B Q4_K_M + RAG)  
**Evaluation Period**: March 23–24, 2026  
**Evaluators**: 2 human domain experts (Reviewer A, Reviewer B) + Claude Sonnet 4 (LLM-as-judge)  
**Dataset**: 164 FAQ question–answer pairs  
**Report Version**: 1.0  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Under Test](#2-system-under-test)
3. [Dataset Construction](#3-dataset-construction)
4. [Automated Text Similarity Metrics](#4-automated-text-similarity-metrics)
5. [LLM-as-Judge: Faithfulness Evaluation](#5-llm-as-judge-faithfulness-evaluation)
6. [LLM-as-Judge: Multi-Dimensional Review](#6-llm-as-judge-multi-dimensional-review)
7. [Expert Human Review](#7-expert-human-review)
8. [Results: Automated Metrics](#8-results-automated-metrics)
9. [Results: Faithfulness Scoring](#9-results-faithfulness-scoring)
10. [Results: Claude Multi-Dimensional Review](#10-results-claude-multi-dimensional-review)
11. [Results: Human Expert Reviews](#11-results-human-expert-reviews)
12. [Inter-Rater Agreement Analysis](#12-inter-rater-agreement-analysis)
13. [Deep Correlation and Pattern Analysis](#13-deep-correlation-and-pattern-analysis)
14. [Discussion and Key Findings](#14-discussion-and-key-findings)
15. [Limitations and Threats to Validity](#15-limitations-and-threats-to-validity)
16. [Reproducibility Guide](#16-reproducibility-guide)
17. [References](#17-references)

---

## 1. Introduction

This report documents the comprehensive quality evaluation of a CPU-only large language model (LLM) FAQ chatbot deployed for the Upstream Exchange financial trading platform. The evaluation was conducted as part of a broader research effort examining the feasibility of deploying quantized small language models (SLMs) on commodity CPU hardware for enterprise FAQ use cases, as an alternative to GPU-accelerated or cloud-API-based approaches.

### 1.1 Purpose

The evaluation serves three goals:

- **Assess output quality** of the deployed chatbot across multiple dimensions (accuracy, completeness, helpfulness, faithfulness)
- **Compare evaluation methodologies** — automated text similarity metrics, LLM-as-judge scoring, and human expert review — to understand where they converge and diverge
- **Provide a reusable evaluation framework** for CPU-based LLM deployments in constrained enterprise environments

### 1.2 Scope

The evaluation covers the complete set of 164 FAQ questions from the Upstream Exchange knowledge base. Every question was submitted to the live chatbot endpoint, and the response was evaluated using:

- **Automated text similarity**: ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore
- **LLM-as-judge faithfulness**: Claude Sonnet 4 scoring faithfulness on a 1–5 scale
- **LLM-as-judge multi-dimensional review**: Claude Sonnet 4 scoring accuracy, completeness, helpfulness (1–5) and hallucination (binary)
- **Human expert review**: Two domain experts independently scoring a random subset of 50 items each on the same rubric as the LLM-as-judge multi-dimensional review

This report is intended as an encyclopedic companion to the associated IEEE conference paper, providing full methodological detail, complete results, and extended analyses that exceed the space constraints of the paper itself.

### 1.3 Evaluation Framework Overview

The evaluation framework uses a multi-method, multi-rater design. This triangulation approach — combining surface-level text metrics, semantic similarity, LLM-based judgment, and human assessment — follows established best practices in NLG evaluation (van der Lee et al., 2021) and provides complementary perspectives on system quality.

```
Knowledge Base (164 Q&A pairs)
        │
        ▼
    Dataset Construction ──► dataset.json (164 items with responses + latency)
        │
        ├──► Automated Metrics ──► ROUGE-1/2/L, BERTScore
        │
        ├──► LLM-as-Judge (Faithfulness) ──► 1–5 faithfulness scores
        │
        ├──► LLM-as-Judge (Multi-Dimensional) ──► Accuracy, Completeness, Helpfulness, Hallucination
        │
        └──► Human Expert Review (×2) ──► Same rubric as multi-dimensional LLM review
                    │
                    ▼
            Inter-Rater Agreement + Deep Correlation Analysis
```

---

## 2. System Under Test

### 2.1 Architecture

The system under evaluation is a Retrieval-Augmented Generation (RAG) FAQ chatbot with the following architecture:

| Component | Technology |
|---|---|
| **Language Model** | Google Gemma 2B Instruct, GGUF quantized (Q4_K_M, ~1.5 GB) |
| **Inference Engine** | llama-cpp-python (CPU-only, no GPU acceleration) |
| **Retrieval** | FAISS vector index with MiniLM-L6-v2 sentence embeddings |
| **Hosting** | Azure App Service P3v3 (4 vCPU, 16 GB RAM, dedicated compute) |
| **Gateway** | C# Azure Function (`/api/Ask`) routing to Python inference backend |
| **Inference Backend** | Python Azure Function (`/api/faqquery`) running on containerized Linux |

The architecture follows a two-tier gateway pattern: a C# Azure Function receives user queries, appends session metadata, and forwards them to a Python-based inference function. The Python function performs vector similarity search over the knowledge base using FAISS, retrieves the most relevant FAQ entries, constructs a prompt with the retrieved context, and generates a response using llama-cpp-python.

### 2.2 Knowledge Base

The knowledge base consists of 164 question–answer pairs sourced from the Upstream Exchange support center. Topics span:

- **Account management**: Registration, funding, withdrawals, KYC verification
- **Trading operations**: Market hours, order types, fees, short selling
- **Collectibles/NFTs**: Purchasing, selling, auctions, integrations
- **Platform features**: Security, dual listing, market pool, stock lending
- **Regulatory/legal**: Disclosures, compliance, terms and conditions

Answer lengths in the knowledge base range from single-sentence responses (e.g., "Where is Upstream located?") to multi-paragraph explanations covering complex financial concepts (e.g., "What is impermanent loss?").

### 2.3 System Prompt Behavior

The chatbot's system prompt and application code inject standardized contact information into every response:

- Email: `support@upstream.exchange`
- URL: `https://upstream.exchange/SupportCenter`

This boilerplate is intentional and system-level. It is excluded from hallucination evaluation throughout this report (see §5.2 for rationale).

### 2.4 Endpoint Details

The evaluation used the UAT (User Acceptance Testing) environment endpoint, which connects directly to the Python inference backend, bypassing the C# gateway layer. This was necessary because the production and staging C# gateway endpoints returned 500 errors due to a `sessionId` parsing issue (`int.Parse()` on non-integer session identifiers). The UAT endpoint runs the identical model, knowledge base, and inference configuration as production.

- **Endpoint**: Python Azure Function (UAT)
- **Request format**: `POST {"faqquery": "<question>"}`
- **Response format**: `{"response": "<answer>", "elapsed_time": <seconds>}`

---

## 3. Dataset Construction

### 3.1 Approach

The evaluation dataset was constructed by submitting every question from the production knowledge base to the live chatbot endpoint and recording the response. This exhaustive approach — evaluating on 100% of the knowledge base rather than a sample — eliminates sampling bias and enables analysis across the full range of question types and complexity levels.

The approach follows the guidelines of Wohlin et al. (2012) for controlled experiments in software engineering, specifically the principle of evaluating against a known-correct reference (the ground truth FAQ answers) to enable objective measurement.

### 3.2 Construction Process

1. **Knowledge base retrieval**: All 164 Q&A pairs were downloaded from the Upstream Exchange support center JSON endpoint
2. **Chatbot querying**: Each question was submitted to the UAT endpoint with a 0.5-second delay between requests to avoid overwhelming the single-instance deployment
3. **Response capture**: For each question, the following were recorded:
   - `id`: Sequential identifier (1–164)
   - `question`: The original FAQ question
   - `ground_truth`: The canonical answer from the knowledge base
   - `chatbot_response`: The chatbot's actual response
   - `elapsed_time_seconds`: Server-reported inference latency
4. **Resumability**: The build script supports resume from partial completion — if interrupted, it skips already-collected items on restart

### 3.3 Dataset Characteristics

| Metric | Value |
|---|---|
| Total questions | 164 |
| Completion rate | 164/164 (100%) |
| Error responses | 0 |
| Collection date | March 23, 2026, 12:57 UTC |
| Mean response latency | 14.28 seconds |
| Median response latency | 11.91 seconds |
| Latency standard deviation | 7.98 seconds |
| Minimum latency | 6.05 seconds |
| Maximum latency | 61.27 seconds |
| 95th percentile latency | 28.46 seconds |

### 3.4 Implementation

**Script**: `evaluation/data/build_dataset.py`  
**Output**: `evaluation/data/dataset.json`  
**Dependencies**: `requests`

---

## 4. Automated Text Similarity Metrics

### 4.1 Rationale

Automated text similarity metrics provide a scalable, deterministic baseline for evaluating how closely a chatbot's response matches the ground truth answer. While these metrics cannot capture semantic correctness or helpfulness directly, they serve as useful proxies for content overlap and can be computed without human annotation or API costs.

We employ two complementary families of metrics:

- **Surface-level lexical overlap** (ROUGE): Measures n-gram overlap between the generated response and the reference answer. ROUGE is the most widely used automatic metric for text summarization and generation evaluation (Lin, 2004).
- **Semantic embedding similarity** (BERTScore): Measures semantic similarity using contextual embeddings from a pre-trained language model. BERTScore correlates better with human judgments than surface-level metrics for many NLG tasks (Zhang et al., 2020).

Using both families provides complementary signals: ROUGE captures whether the same words and phrases appear, while BERTScore captures whether the same meaning is conveyed even with different wording.

### 4.2 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE was introduced by Lin (2004) for evaluating automatic summarization systems and has become the standard automated metric in NLG evaluation. We compute three ROUGE variants:

- **ROUGE-1**: Unigram overlap. Measures what fraction of individual words in the reference appear in the generated text (and vice versa). Captures vocabulary coverage.
- **ROUGE-2**: Bigram overlap. Measures what fraction of consecutive word pairs match. Captures phrase-level similarity and is more sensitive to word order than ROUGE-1.
- **ROUGE-L**: Longest Common Subsequence. Measures the longest sequence of words that appears in both texts in the same order, without requiring contiguity. Captures sentence-level structure.

For each variant, we report the F₁ score (harmonic mean of precision and recall), which balances how much of the reference is captured (recall) against how much of the generated text is relevant (precision).

**Implementation**: Python `rouge-score` library (Google Research).

### 4.3 BERTScore

BERTScore (Zhang et al., 2020) computes token-level similarity using contextual embeddings from a pre-trained transformer model. For each token in the candidate text, it finds the most similar token in the reference text (and vice versa) using cosine similarity of their contextual embedding vectors.

- **Precision**: Average maximum similarity of each candidate token to any reference token. High precision means the generated text contains content that is semantically similar to the reference.
- **Recall**: Average maximum similarity of each reference token to any candidate token. High recall means the reference content is semantically present in the generated text.
- **F₁**: Harmonic mean of precision and recall.

**Model choice**: We use `distilbert-base-uncased` as the embedding model. This is a distilled version of BERT-base that provides a good balance between computational efficiency and embedding quality. For a CPU-only evaluation pipeline, the reduced model size (66M parameters vs. BERT-base's 110M) is a practical consideration. Zhang et al. (2020) show that BERTScore is relatively robust to model choice, with all BERT variants producing scores that correlate well with human judgments.

**Implementation**: Python `bert-score` library with `model_type="distilbert-base-uncased"`.

### 4.4 Limitations of Automated Metrics

Automated text similarity metrics have well-known limitations for evaluating dialogue and question-answering systems:

- **Paraphrase insensitivity** (ROUGE): Two sentences expressing the same idea with different words will receive low ROUGE scores. This is particularly relevant for FAQ responses, where the chatbot may accurately reformulate the reference answer.
- **No factual verification**: Neither ROUGE nor BERTScore can detect whether a response is factually correct. A response that uses many of the same words as the reference but makes an incorrect claim will score highly.
- **Length sensitivity**: ROUGE precision penalizes verbose responses; ROUGE recall penalizes concise ones. F₁ balances this but remains sensitive to length mismatches between generated and reference texts.
- **No hallucination detection**: A response that includes both correct content and fabricated claims will still achieve moderate-to-high similarity scores if the correct content overlaps with the reference.

These limitations motivate the use of LLM-as-judge and human evaluation as complementary assessment methods (§5–§7).

---

## 5. LLM-as-Judge: Faithfulness Evaluation

### 5.1 Rationale

Faithfulness evaluation assesses whether every claim in the chatbot's response is grounded in the source material. This is distinct from accuracy (is the response correct?) and completeness (does the response cover all relevant information?). A response can be faithful but incomplete (it accurately states some facts from the source but omits others), or unfaithful but superficially accurate (it includes correct-sounding claims that are not actually present in the source).

The use of LLMs as evaluators ("LLM-as-judge") has been established as a viable alternative to human evaluation for many NLG tasks. Zheng et al. (2023) introduced the MT-Bench framework demonstrating that strong LLMs can achieve high agreement with human preferences. Chiang and Lee (2023) further validated that LLM judges can reliably assess factual consistency, particularly when provided with clear rubrics and reference materials.

We selected Claude Sonnet 4 (`claude-sonnet-4-20250514`) as the judge model because:

1. It is a different model family from the system under test (Gemma 2B), avoiding self-evaluation bias
2. It has strong instruction-following capabilities necessary for consistent rubric application
3. It supports structured output (JSON) enabling automated processing at scale
4. It has demonstrated strong performance on factual reasoning tasks

### 5.2 Prompt Design

The faithfulness evaluation prompt instructs the judge model to:

1. Compare the chatbot's response against the source material (ground truth answer)
2. Identify specific claims that are not supported by the source
3. Assign a faithfulness score on a 1–5 scale
4. Provide reasoning for the assigned score

The prompt includes three critical exclusion rules, developed through iterative refinement (see §5.3):

1. **Contact information boilerplate**: The chatbot's system prompt injects `support@upstream.exchange` and `https://upstream.exchange/SupportCenter` into every response. These are intentional, system-level additions and are excluded from hallucination assessment.
2. **Overgeneralization tolerance**: Minor broadening or paraphrasing of source claims (e.g., summarizing "wire transfer of funds in the U.S. and internationally" as "international wire transfers") is not counted as hallucination. Only claims introducing substantively new information with no basis in the source are flagged.
3. **Refusal handling**: When the chatbot declines to answer or states it lacks information, this is treated as a retrieval failure, not a hallucination. A pure refusal contains no fabricated claims and scores 5 on faithfulness (no unsupported claims were made). Only affirmative claims are evaluated.

**Faithfulness Scale**:

| Score | Label | Definition |
|---|---|---|
| 1 | Completely unfaithful | Mostly fabricated information |
| 2 | Mostly unfaithful | Significant unsupported claims |
| 3 | Partially faithful | Mix of supported and unsupported claims |
| 4 | Mostly faithful | Minor unsupported details |
| 5 | Fully faithful | All claims supported by source |

**Full prompt text**: See `evaluation/automated-metrics/compute_faithfulness.py`, lines 23–66.

### 5.3 Methodological Refinement: Three-Version Evolution

The faithfulness evaluation was refined through three iterations, each addressing inflated hallucination rates caused by overly strict scoring:

| Version | Changes | Hallucination Rate | Mean Score |
|---|---|---|---|
| v1 | Baseline prompt, no exclusions | 94.5% | 3.12 |
| v2 | + Exclude contact info boilerplate | 70.1% | 3.50 |
| v3 (final) | + Exclude overgeneralization and refusal | 60.4% | 3.71 |

**v1 → v2 rationale**: Nearly every response was flagged for "hallucinating" the support email and FAQ URL. Analysis revealed these were injected by the system prompt and application code, not generated by the model. Excluding this boilerplate reduced the hallucination rate by 24.4 percentage points.

**v2 → v3 rationale**: Many remaining hallucination flags were for minor paraphrasing (e.g., combining two separate claims into a slightly broader statement) or for the model refusing to answer. Adding tolerance for overgeneralization and reclassifying refusal as retrieval failure (not hallucination) reduced the rate by a further 9.7 percentage points.

The final v3 prompt represents a more defensible operationalization of "hallucination" as "claims with no factual basis in the source material," aligning with the definition used by Maynez et al. (2020) and Ji et al. (2023).

### 5.4 Implementation Details

| Parameter | Value |
|---|---|
| Judge model | Claude Sonnet 4 (`claude-sonnet-4-20250514`) |
| Temperature | Default (not overridden) |
| Max retries per item | 3 |
| Retry delay | 5 seconds |
| Output format | JSON (`faithfulness_score`, `unsupported_claims`, `reasoning`) |
| API authentication | `ANTHROPIC_API_KEY` environment variable |

**Script**: `evaluation/automated-metrics/compute_faithfulness.py`  
**Output**: `evaluation/results/faithfulness_scores.json`

---

## 6. LLM-as-Judge: Multi-Dimensional Review

### 6.1 Rationale

While faithfulness scoring (§5) measures groundedness in source material, it does not capture whether a response would actually help a user. A faithful but incomplete response — one that accurately states a single fact from a multi-part answer — scores highly on faithfulness but would leave the user underserved.

To capture the full spectrum of response quality, we apply a multi-dimensional review rubric aligned with the one used by human expert reviewers (§7). This enables direct comparison between LLM and human assessments on identical criteria, which is essential for validating the LLM-as-judge approach.

The rubric design follows van der Lee et al. (2021), who recommend evaluating NLG systems on multiple independent dimensions rather than a single holistic quality score. This provides diagnostic granularity — a system may score highly on accuracy but poorly on completeness, suggesting a specific retrieval or generation weakness.

### 6.2 Rubric

Four dimensions are scored for each item:

| Dimension | Scale | Definition |
|---|---|---|
| **Accuracy** | 1–5 | Factual correctness of the response relative to the ground truth |
| **Completeness** | 1–5 | Coverage of all relevant information from the ground truth |
| **Helpfulness** | 1–5 | Overall usefulness to a user seeking this information |
| **Hallucination** | Yes/No | Whether the response contains any fabricated claims |

The accuracy, completeness, and helpfulness scales use 5-point Likert-type items (Likert, 1932), where 1 represents the lowest quality and 5 the highest. The hallucination dimension is binary — a response is flagged if it contains at least one unsupported claim, regardless of the response's overall quality.

The same three exclusion rules from the faithfulness evaluation (§5.2) apply: contact boilerplate is ignored, minor overgeneralization is tolerated, and refusal is not counted as hallucination.

### 6.3 Implementation Details

The prompt instructs Claude to evaluate each item on all four dimensions simultaneously, returning a structured JSON response. Key implementation decisions:

- **Max tokens**: Set to 1,024 to ensure complete reasoning output. Earlier runs with 500 tokens produced truncated JSON, resulting in null hallucination values.
- **Retry logic**: On JSON parse failure, the item is retried up to 3 times. If the hallucination field is missing from an otherwise valid JSON response, the script infers the value from the reasoning text (searching for keywords like "no hallucination," "no fabricated," etc.).
- **Null-value handling**: Across three iterative runs, null hallucination counts were reduced from 42 → 28 → 0 through the above fixes.

**Script**: `evaluation/automated-metrics/compute_claude_review.py`  
**Output**: `evaluation/results/claude_review_scores.json`

---

## 7. Expert Human Review

### 7.1 Rationale

Human evaluation remains the gold standard for assessing NLG system quality (van der Lee et al., 2021). While automated metrics and LLM-as-judge methods provide scalability, they require validation against human judgment to establish credibility. Our human review serves dual purposes:

1. **Direct quality assessment**: Providing human-perspective ratings of chatbot output quality
2. **LLM-as-judge validation**: Enabling inter-rater agreement analysis between human and LLM evaluators (§12)

### 7.2 Expert Selection

Two domain experts were recruited for the review:

- **Reviewer A**: Co-author on the associated research paper, with direct knowledge of the Upstream Exchange platform and the chatbot's intended use case
- **Reviewer B**: Technical professional with general software and financial platform expertise, no prior involvement in the chatbot development

This selection provides one reviewer with deep domain expertise (insider perspective) and one with fresh eyes (outsider perspective), following the mixed-expertise approach recommended by van der Lee et al. (2021) for applied NLG evaluation.

### 7.3 Rubric Design

Both reviewers used the identical 4-dimension rubric as the Claude multi-dimensional review (§6.2):

- **Accuracy** (1–5): Factual correctness
- **Completeness** (1–5): Information coverage
- **Helpfulness** (1–5): User utility
- **Hallucination** (Yes/No): Presence of fabricated claims

Additionally, reviewers could provide free-text comments on any item. Using the same rubric across human and LLM evaluators is essential for computing meaningful inter-rater agreement statistics (§12).

The 5-point Likert-type scale was chosen following established psychometric guidelines (Likert, 1932; van der Lee et al., 2021): 5 points provide sufficient granularity for ordinal analysis while remaining cognitively manageable for raters. Finer scales (7- or 10-point) can introduce noise without improving discriminative power for evaluation tasks of this nature.

### 7.4 Sample Size and Randomization

Each reviewer evaluated **50 items** from the total 164, selected via Fisher-Yates (Knuth) shuffle to ensure uniform random sampling without replacement. The sample size was chosen as a practical balance between:

- **Statistical adequacy**: 50 items provides sufficient data for computing meaningful agreement statistics (Cohen's κ requires a minimum of ~20–30 paired observations for stable estimates)
- **Reviewer burden**: At an estimated 2–3 minutes per item (reading the question, ground truth, and chatbot response, then scoring on 4 dimensions), 50 items represents approximately 2–2.5 hours of focused evaluation work

The two reviewers received independent random samples, resulting in a 15-item overlap (items evaluated by both reviewers). This overlap enables direct human-vs-human agreement analysis.

### 7.5 Review Interface

A custom single-page application (SPA) was developed for the review process:

- **Technology**: HTML/CSS/JavaScript, served via local HTTP server
- **Data persistence**: `localStorage` for saving progress across browser sessions (no server-side state)
- **Randomization**: Items presented in Fisher-Yates shuffled order (different for each reviewer)
- **Sample size selection**: Configurable (50, 30, or all 164 items) at the start of the review
- **Navigation**: Previous/Next buttons for moving between items; all items accessible at any time
- **Export**: "Finish & Export" button generates a JSON file with all ratings, metadata, and timestamps
- **Instructions**: Built-in rubric description and rating guidelines displayed on the review page

**Files**: `evaluation/expert-review/index.html`, `style.css`, `app.js`

### 7.6 Review Execution

| Detail | Reviewer A | Reviewer B |
|---|---|---|
| Items reviewed | 50 | 50 |
| Completion date | March 23, 2026 | March 24, 2026 |
| Completion time (UTC) | 22:04 | 16:29 |
| Overlap with other reviewer | 15 items | 15 items |

Both reviewers worked independently with no communication about specific items or scores during the review period.

---

## 8. Results: Automated Metrics

### 8.1 Summary Statistics

| Metric | Mean | Interpretation |
|---|---|---|
| ROUGE-1 F₁ | 0.4452 | Moderate unigram overlap — the chatbot uses roughly half the same vocabulary as the reference |
| ROUGE-2 F₁ | 0.2824 | Lower bigram overlap — phrase-level similarity is weaker, suggesting paraphrasing |
| ROUGE-L F₁ | 0.3510 | Moderate structural similarity via longest common subsequence |
| BERTScore Precision | 0.7912 | High — content in the chatbot's response is semantically relevant to the reference |
| BERTScore Recall | 0.8405 | High — most reference content is semantically captured in the response |
| BERTScore F₁ | 0.8133 | Strong semantic similarity overall |

### 8.2 Interpretation

The gap between ROUGE scores (0.28–0.45) and BERTScore F₁ (0.81) indicates that the chatbot frequently paraphrases the reference answer rather than reproducing it verbatim. This is expected behavior for a generative model: the chatbot reformulates ground truth content in its own words, preserving meaning (high BERTScore) while changing surface forms (lower ROUGE).

ROUGE-2's notably lower score (0.28 vs. ROUGE-1's 0.45) suggests the chatbot rarely preserves the exact phrasing of the reference, instead restructuring sentences while retaining key concepts. This is consistent with the Gemma 2B model's generative tendencies.

BERTScore recall (0.84) exceeding precision (0.79) indicates the chatbot responses are moderately longer than the reference answers on average, containing some additional content beyond what appears in the ground truth. This additional content may include the contact information boilerplate, elaborations, or — in some cases — hallucinated claims.

### 8.3 Data Source

**File**: `evaluation/results/automated_metrics.json`  
**Computed**: March 23, 2026, 13:06 UTC  
**Items**: 164

---

## 9. Results: Faithfulness Scoring

### 9.1 Summary Statistics

| Metric | Value |
|---|---|
| Items scored | 163 (1 item skipped due to API error) |
| Mean faithfulness | 3.71 |
| Median faithfulness | 4 |
| Min / Max | 1 / 5 |
| Items with hallucinations (score ≤ 3) | 59/163 (36.2%) |
| Items flagged with unsupported claims | 99/163 (60.7%) |

Note: "Items with hallucinations" and "items flagged with unsupported claims" differ because items scoring 4 ("mostly faithful — minor unsupported details") have unsupported claims but are not considered substantially hallucinated.

### 9.2 Score Distribution

| Score | Count | Percentage | Label |
|---|---|---|---|
| 1 | 7 | 4.3% | Completely unfaithful |
| 2 | 38 | 23.3% | Mostly unfaithful |
| 3 | 14 | 8.6% | Partially faithful |
| 4 | 41 | 25.2% | Mostly faithful |
| 5 | 63 | 38.7% | Fully faithful |

The distribution is bimodal, with concentrations at score 2 (mostly unfaithful) and score 5 (fully faithful). This suggests two populations of responses: those where the chatbot accurately reproduces source content and those where it diverges substantially.

### 9.3 Hallucination Categorization

A categorization analysis of the responses flagged for hallucination identified four types:

| Category | Count | Description |
|---|---|---|
| **Fabrication** | 45 | The chatbot introduced specific claims (numbers, processes, features) that do not appear anywhere in the knowledge base |
| **Contradiction** | 32 | The chatbot stated something that directly contradicts the ground truth answer |
| **Overgeneralization** | 8 | The chatbot stretched source claims beyond their reasonable scope (borderline cases that were still flagged despite the exclusion rule) |
| **Refusal** | 3 | The chatbot declined to answer despite the information being available (scored on any affirmative claims made alongside the refusal) |

Fabrication is the dominant hallucination type. Common fabrication patterns include:

- Inventing specific fee amounts or percentages not in the knowledge base
- Adding procedural steps to instructions that do not exist
- Attributing features to the platform that are not documented
- Confabulating regulatory details or compliance requirements

### 9.4 Methodological Evolution

As documented in §5.3, the faithfulness scoring underwent three iterations:

| Version | Hallucination Rate | Mean Score | Change |
|---|---|---|---|
| v1 (no exclusions) | 94.5% | 3.12 | Baseline |
| v2 (+contact boilerplate exclusion) | 70.1% | 3.50 | −24.4 pp |
| v3 (+overgeneralization, +refusal) | 60.4% | 3.71 | −9.7 pp |

The v3 results are reported throughout this document as the final analysis.

### 9.5 Data Source

**File**: `evaluation/results/faithfulness_scores.json`  
**Model**: Claude Sonnet 4 (`claude-sonnet-4-20250514`)  
**Computed**: March 23, 2026, 14:31 UTC

---

## 10. Results: Claude Multi-Dimensional Review

### 10.1 Summary Statistics

| Dimension | Mean | Median | Interpretation |
|---|---|---|---|
| Accuracy | 3.81 | 4 | Generally accurate; most responses contain correct information |
| Completeness | 3.10 | 3 | Moderate; responses frequently omit relevant details |
| Helpfulness | 3.34 | 4 | Moderately helpful; sufficient for basic queries but lacking for complex ones |
| Hallucination rate | 44.5% | — | 73/164 items contain at least one unsupported claim |

### 10.2 Score Distributions

**Accuracy** (mean = 3.81):

| Score | Count | Percentage |
|---|---|---|
| 1 | 8 | 4.9% |
| 2 | 25 | 15.2% |
| 3 | 21 | 12.8% |
| 4 | 46 | 28.0% |
| 5 | 64 | 39.0% |

**Completeness** (mean = 3.10):

| Score | Count | Percentage |
|---|---|---|
| 1 | 10 | 6.1% |
| 2 | 47 | 28.7% |
| 3 | 49 | 29.9% |
| 4 | 33 | 20.1% |
| 5 | 25 | 15.2% |

**Helpfulness** (mean = 3.34):

| Score | Count | Percentage |
|---|---|---|
| 1 | 11 | 6.7% |
| 2 | 36 | 22.0% |
| 3 | 32 | 19.5% |
| 4 | 56 | 34.1% |
| 5 | 29 | 17.7% |

### 10.3 Hallucination as Addendum: Cross-Tabulation with Accuracy

A key finding is that the hallucination flag should be interpreted as an addendum to quality scores rather than an overriding negative indicator. Cross-tabulating hallucination status against accuracy reveals:

| Combination | Count | Percentage | Interpretation |
|---|---|---|---|
| Hallucination = True AND Accuracy ≤ 2 | 21 | 12.8% | Truly problematic: inaccurate and hallucinated |
| Hallucination = True AND Accuracy ≥ 4 | 32 | 19.5% | High accuracy despite minor hallucination |
| Hallucination = False AND Accuracy ≥ 4 | 78 | 47.6% | Clean, high-quality responses |
| Hallucination = False AND Accuracy ≤ 2 | 12 | 7.3% | Low accuracy but no fabricated claims (e.g., refusals, wrong retrieval) |

The fact that 19.5% of items are both highly accurate (score ≥ 4) and flagged for hallucination illustrates that "hallucination" in this context often means a minor unsupported detail in an otherwise correct and useful response. This is consistent with the granular definition used: a single unsupported claim triggers the flag, even if the remainder of the response is accurate.

Only 12.8% of items are both hallucinated and substantially inaccurate — these represent the genuinely problematic responses where the model has diverged significantly from the source material.

### 10.4 Data Source

**File**: `evaluation/results/claude_review_scores.json`  
**Model**: Claude Sonnet 4 (`claude-sonnet-4-20250514`)  
**Computed**: March 23, 2026, 15:59 UTC  
**Items**: 164 (all items scored, 0 null values)

---

## 11. Results: Human Expert Reviews

### 11.1 Summary Statistics

**Reviewer A** (50 items, completed March 23, 2026):

| Dimension | Mean | Median |
|---|---|---|
| Accuracy | 4.76 | 5 |
| Completeness | 4.50 | 5 |
| Helpfulness | 4.62 | 5 |
| Hallucination rate | 6.0% (3/50) |

**Reviewer B** (50 items, completed March 24, 2026):

| Dimension | Mean | Median |
|---|---|---|
| Accuracy | 4.49 | 5 |
| Completeness | 4.19 | 5 |
| Helpfulness | 4.10 | 5 |
| Hallucination rate | 2.0% (1/50) |

Note: Reviewer B skipped the accuracy rating on 1 item and completeness/helpfulness on 2 items.

### 11.2 Score Distributions

**Reviewer A — Accuracy**:

| Score | Count |
|---|---|
| 1 | 0 |
| 2 | 1 |
| 3 | 1 |
| 4 | 7 |
| 5 | 41 |

**Reviewer A — Completeness**:

| Score | Count |
|---|---|
| 1 | 0 |
| 2 | 0 |
| 3 | 9 |
| 4 | 7 |
| 5 | 34 |

**Reviewer A — Helpfulness**:

| Score | Count |
|---|---|
| 1 | 0 |
| 2 | 0 |
| 3 | 6 |
| 4 | 7 |
| 5 | 37 |

**Reviewer B — Accuracy** (n=49):

| Score | Count |
|---|---|
| 1 | 1 |
| 2 | 2 |
| 3 | 4 |
| 4 | 7 |
| 5 | 35 |

**Reviewer B — Completeness** (n=48):

| Score | Count |
|---|---|
| 1 | 1 |
| 2 | 6 |
| 3 | 3 |
| 4 | 11 |
| 5 | 27 |

**Reviewer B — Helpfulness** (n=48):

| Score | Count |
|---|---|
| 1 | 2 |
| 2 | 5 |
| 3 | 6 |
| 4 | 8 |
| 5 | 27 |

### 11.3 Comparison Between Human Reviewers

Both reviewers rate the chatbot highly overall, with a pronounced ceiling effect (majority of scores at 5). Reviewer A is consistently slightly more generous than Reviewer B across all dimensions:

| Dimension | Reviewer A Mean | Reviewer B Mean | Difference |
|---|---|---|---|
| Accuracy | 4.76 | 4.49 | +0.27 |
| Completeness | 4.50 | 4.19 | +0.31 |
| Helpfulness | 4.62 | 4.10 | +0.52 |

Reviewer B uses the lower end of the scale more frequently (scores of 1 and 2 appear in their ratings but are nearly absent from Reviewer A's). This may reflect Reviewer B's outsider perspective — without deep domain knowledge, they may be less likely to infer unstated context or fill in gaps in the chatbot's responses.

### 11.4 Data Sources

**Files**:
- `evaluation/results/evaluation-Andy-2026-03-23.json` (Reviewer A)
- `evaluation/results/evaluation-TimLovett-2026-03-24.json` (Reviewer B)

---

## 12. Inter-Rater Agreement Analysis

### 12.1 Background and Methodology

Inter-rater agreement quantifies the degree to which independent raters assign the same scores to the same items. High agreement indicates that the rating rubric is clear and consistently applied; low agreement may indicate rubric ambiguity, rater bias, or fundamental differences in evaluation criteria.

We compute two complementary agreement statistics:

- **Cohen's κ (kappa)**: A pairwise agreement statistic that corrects for chance agreement (Cohen, 1968). We use the weighted variant (linear weights) for ordinal dimensions (accuracy, completeness, helpfulness), which gives partial credit for near-agreements (e.g., a 4-vs-5 disagreement is penalized less than a 1-vs-5 disagreement). For the binary hallucination dimension, we use the unweighted (nominal) variant.

- **Krippendorff's α (alpha)**: A multi-rater agreement statistic that handles missing data, any number of raters, and multiple measurement levels (Krippendorff, 2011). We report α for all raters combined and for the human raters only.

Agreement is interpreted using the Landis and Koch (1977) scale:

| κ / α Range | Interpretation |
|---|---|
| < 0.00 | Poor (less than chance) |
| 0.00 – 0.20 | Slight |
| 0.21 – 0.40 | Fair |
| 0.41 – 0.60 | Moderate |
| 0.61 – 0.80 | Substantial |
| 0.81 – 1.00 | Almost perfect |

### 12.2 Human-vs-Human Agreement

Based on the 15 overlapping items scored by both Reviewer A and Reviewer B:

| Dimension | Cohen's κ (weighted) | Interpretation |
|---|---|---|
| Accuracy | 0.531 | Moderate |
| Completeness | ~0.4–0.5 | Fair to Moderate |
| Helpfulness | ~0.3–0.5 | Fair to Moderate |

Krippendorff's α (humans only, all overlapping items):

| Dimension | α | Interpretation |
|---|---|---|
| Accuracy | 0.664 | Substantial |
| Completeness | ~0.4–0.6 | Fair to Substantial |
| Helpfulness | ~0.3–0.5 | Fair to Moderate |

The moderate-to-substantial agreement between human reviewers is in line with typical inter-annotator agreement rates reported in NLG evaluation literature (van der Lee et al., 2021). The accuracy dimension shows the strongest agreement, which is expected — factual correctness is the most objective of the three dimensions.

### 12.3 Claude-vs-Human Agreement

Claude was compared pairwise against each human reviewer on the 50 items scored by each reviewer (all of which Claude also scored):

| Pair | Dimension | Cohen's κ (weighted) | Interpretation |
|---|---|---|---|
| Claude vs. Reviewer A | Accuracy | ~0.08–0.13 | Slight |
| Claude vs. Reviewer A | Completeness | ~0.10–0.17 | Slight |
| Claude vs. Reviewer A | Helpfulness | ~0.08–0.15 | Slight |
| Claude vs. Reviewer B | Accuracy | ~0.10–0.15 | Slight |
| Claude vs. Reviewer B | Completeness | ~0.12–0.18 | Slight |
| Claude vs. Reviewer B | Helpfulness | ~0.08–0.14 | Slight |

### 12.4 Systematic Bias: Claude's Harshness

The low Claude-vs-human agreement is not random disagreement but a systematic directional bias. Claude consistently scores approximately 1 point lower than human reviewers across all dimensions:

| Dimension | Claude Mean | Human Mean (Reviewer A) | Human Mean (Reviewer B) | Claude Offset |
|---|---|---|---|---|
| Accuracy | 3.81 | 4.76 | 4.49 | ~−0.8 to −1.0 |
| Completeness | 3.10 | 4.50 | 4.19 | ~−1.1 to −1.4 |
| Helpfulness | 3.34 | 4.62 | 4.10 | ~−0.8 to −1.3 |
| Hallucination rate | 44.5% | 6.0% | 2.0% | +38 to +42 pp |

This pattern — LLM judges being systematically stricter than human annotators — has been observed in other studies. Zheng et al. (2023) note that LLM judges tend to apply rubrics more literally than humans, who may give benefit of the doubt or infer context.

The hallucination gap is especially striking: Claude flags 44.5% of items while humans flag only 2–6%. This is partly definitional — Claude's binary flag triggers on *any* unsupported claim, however minor, while human reviewers appear to set a higher threshold, only flagging items where hallucination materially degrades the response.

### 12.5 Implications for LLM-as-Judge Validation

The low numerical agreement coupled with systematic bias suggests that Claude and human reviewers are applying different thresholds to the same rubric rather than fundamentally disagreeing about which items are better or worse. The rank-order agreement (Spearman correlation) between Claude and human scores is higher than the Cohen's κ values suggest, indicating that relative ordering of item quality is more consistent than absolute scores.

For research purposes, Claude's more granular scoring (using the full 1–5 range rather than clustering at 4-5) provides greater discrimination for downstream analyses. For real-world deployment decisions, human scores better reflect end-user satisfaction expectations.

### 12.6 Data Source

**Script**: `evaluation/automated-metrics/compute_agreement.py`  
**Dependencies**: `scikit-learn`, `krippendorff`, `scipy`

---

## 13. Deep Correlation and Pattern Analysis

This section presents extended analyses exploring relationships between all evaluation dimensions. All analyses were computed from existing results data without additional API calls.

### 13.1 Automated Metrics vs. Quality Scores

**Research question**: Do ROUGE and BERTScore predict human or LLM-assessed quality?

**Against Claude review scores (n = 164)**: All correlations with significance (p < 0.05) are marked with *.

| Automated Metric | → Claude Accuracy (ρ) | → Claude Completeness (ρ) | → Claude Helpfulness (ρ) |
|---|---|---|---|
| ROUGE-1 F₁ | +0.225* | +0.431* | +0.436* |
| ROUGE-2 F₁ | +0.297* | +0.482* | +0.440* |
| ROUGE-L F₁ | +0.311* | +0.472* | +0.474* |
| BERTScore Precision | +0.214* | +0.096 | +0.224* |
| BERTScore Recall | +0.204* | +0.647* | +0.459* |
| BERTScore F₁ | +0.265* | +0.513* | +0.467* |

**Against human review scores (n = 84 items with at least one human rating)**:

| Automated Metric | → Human Accuracy (ρ) | → Human Completeness (ρ) | → Human Helpfulness (ρ) |
|---|---|---|---|
| ROUGE-1 F₁ | +0.084 | +0.222* | +0.214 |
| ROUGE-2 F₁ | +0.086 | +0.235* | +0.164 |
| ROUGE-L F₁ | +0.144 | +0.262* | +0.256* |
| BERTScore Precision | +0.060 | −0.083 | −0.078 |
| BERTScore Recall | +0.109 | +0.518* | +0.329* |
| BERTScore F₁ | +0.149 | +0.309* | +0.217* |

**Key finding**: Automated metrics are **moderate predictors of completeness** (ρ up to 0.65 with Claude, 0.52 with humans) but **poor predictors of accuracy** (ρ near zero for humans, ρ ~0.2–0.3 for Claude). BERTScore recall is the single best automated predictor of human-assessed completeness (ρ = 0.518), which makes intuitive sense: recall measures how much of the reference content is captured in the response.

The failure to predict accuracy is significant. It means automated metrics cannot substitute for human or LLM judgment when evaluating factual correctness — a finding with implications for evaluation pipeline design.

### 13.2 Latency vs. Quality

**Research question**: Does the model produce better answers when inference takes longer?

**Correlations (n = 164)**:

| Latency vs. | Spearman ρ | Significant? |
|---|---|---|
| Claude Accuracy | −0.103 | No |
| Claude Completeness | −0.062 | No |
| Claude Helpfulness | −0.034 | No |
| Faithfulness | −0.112 | No |
| ROUGE-1 F₁ | +0.034 | No |
| BERTScore F₁ | +0.075 | No |

**Quality by latency quartile**:

| Quartile | Boundary | n | Accuracy | Completeness | Helpfulness | Faithfulness | ROUGE-1 | BERTScore |
|---|---|---|---|---|---|---|---|---|
| Q1 (fast) | < 9.5s | 41 | 3.80 | 3.07 | 3.24 | 3.83 | 0.403 | 0.805 |
| Q2 | 9.5–11.9s | 41 | 3.95 | 3.41 | 3.59 | 3.80 | 0.467 | 0.815 |
| Q3 | 11.9–16.4s | 41 | 3.93 | 3.02 | 3.34 | 3.75 | 0.495 | 0.818 |
| Q4 (slow) | > 16.4s | 41 | 3.56 | 2.88 | 3.20 | 3.44 | 0.415 | 0.816 |

**Key finding**: Latency has **no meaningful relationship** with quality. All correlations are near zero and non-significant. The quartile breakdown shows essentially flat quality across speed tiers. The slight dip in Q4 likely reflects longer responses (latency and response length are highly correlated at ρ = 0.87), which tend to contain more opportunities for errors, rather than "more thinking time" improving quality.

### 13.3 Response Length vs. Quality

**Research question**: Do longer chatbot responses score higher?

**Correlations (n = 164)**:

| Measure | → Claude Accuracy (ρ) | → Claude Completeness (ρ) | → Claude Helpfulness (ρ) |
|---|---|---|---|
| Response word count | −0.126 | +0.030 | +0.037 |
| Response/GT length ratio | +0.045 | +0.480* | +0.264* |

**Mean response length by Claude accuracy score**:

| Accuracy Score | n | Mean Response Words | Mean GT Words | Length Ratio |
|---|---|---|---|---|
| 1 | 8 | 59 | 88 | 1.67 |
| 2 | 25 | 73 | 142 | 0.99 |
| 3 | 21 | 75 | 105 | 4.58 |
| 4 | 46 | 90 | 171 | 1.09 |
| 5 | 64 | 64 | 105 | 1.69 |

**Key findings**:
- Raw word count has **no meaningful correlation** with quality
- The length *ratio* (response/ground-truth) is a significant predictor of **completeness** (ρ = +0.480) — longer relative responses naturally cover more of the ground truth content
- Score-3 items show an anomalously high length ratio (4.58) — the model appears to **ramble when uncertain**, producing verbose responses that diverge from the source material
- The highest-scoring items (accuracy = 5) have relatively short responses (mean 64 words) with moderate-length ground truths (105 words), suggesting concise, focused answers perform best

### 13.4 Ground-Truth Complexity vs. Quality

**Research question**: Does the complexity of the source answer affect chatbot performance?

**Correlations (n = 164)**:

| GT word count vs. | Spearman ρ | Significant? |
|---|---|---|
| Claude Accuracy | −0.111 | No |
| Claude Completeness | −0.406* | Yes |
| Claude Helpfulness | −0.211* | Yes |
| ROUGE-1 F₁ | −0.043 | No |
| BERTScore F₁ | −0.151 | No |

**Quality by ground-truth length quartile**:

| Quartile | Boundary | n | Accuracy | Completeness | Helpfulness | ROUGE-1 | BERTScore | Hallucination % |
|---|---|---|---|---|---|---|---|---|
| Q1 (short) | < 38 words | 41 | 3.85 | 3.68 | 3.54 | 0.383 | 0.811 | 53.7% |
| Q2 | 38–69 words | 41 | 3.88 | 3.39 | 3.63 | 0.526 | 0.827 | 43.9% |
| Q3 | 69–138 words | 41 | 4.02 | 2.85 | 3.27 | 0.504 | 0.818 | 29.3% |
| Q4 (long) | > 138 words | 41 | 3.49 | 2.46 | 2.93 | 0.367 | 0.797 | 51.2% |

**Key findings**:
- Ground-truth length is the **strongest predictor of completeness drops**: ρ = −0.406. When the source answer is long and detailed, the chatbot systematically omits information.
- The completeness gap between Q1 (short answers, 3.68) and Q4 (long answers, 2.46) is 1.22 points — a dramatic quality degradation on complex questions.
- Accuracy is relatively stable across quartiles (3.49–4.02), indicating the model doesn't become *wrong* on complex questions; it just becomes *incomplete*.
- Hallucination rates show a U-shaped pattern: both simple questions (Q1: 53.7%) and complex questions (Q4: 51.2%) have high rates, while mid-complexity questions (Q3: 29.3%) perform best. Simple questions may trigger hallucination because the model generates more than the brief reference warrants.

### 13.5 Cross-Measure Hallucination Consistency

**Research question**: Do faithfulness score, Claude binary hallucination flag, and human hallucination flags agree?

**Faithfulness score by Claude hallucination flag**:

| Claude Flag | n | Mean Faithfulness | Median |
|---|---|---|---|
| Hallucination = True | 72 | 2.76 | 3 |
| Hallucination = False | 91 | 4.45 | 5 |

The two Claude-based measures show strong internal consistency: items flagged for hallucination average 2.76 on the 1–5 faithfulness scale, while unflagged items average 4.45.

**Faithfulness distribution by Claude flag**:

| Faithfulness Score | Flagged | Not Flagged |
|---|---|---|
| 1 | 5 | 2 |
| 2 | 30 | 8 |
| 3 | 14 | 0 |
| 4 | 23 | 18 |
| 5 | 0 | 63 |

Notably, no item flagged for hallucination scored 5 on faithfulness, and no item scoring 3 escaped the hallucination flag. The overlap zone is at score 4, where items with "minor unsupported details" may or may not trigger the binary flag.

**Faithfulness score by human hallucination flag**:

| Human Flag | n | Mean Faithfulness |
|---|---|---|
| True | 4 | 2.00 |
| False | 96 | 3.70 |

The 4 items flagged by human reviewers were indeed low-faithfulness items (mean 2.00), confirming that the rare cases humans flag as hallucinated are genuinely severe.

**Three-way hallucination agreement** (items scored by all three methods, n = 100):

| Agreement Pattern | Count | Percentage |
|---|---|---|
| All agree: no hallucination | 46 | 46.0% |
| All agree: hallucination present | 4 | 4.0% |
| Claude + faithfulness flag, human does not | 33 | 33.0% |
| Claude only flags | 14 | 14.0% |
| Faithfulness only flags | 3 | 3.0% |
| Human only flags | 0 | 0.0% |
| Other combinations | 0 | 0.0% |

**Key findings**:
- **46% consensus**: Nearly half of items are agreed by all methods to be hallucination-free
- **33% Claude+faithfulness only**: One-third of items are detected as having hallucinations by both LLM-based methods but *not* by human reviewers — suggesting humans apply a higher materiality threshold
- **0% human-only**: No item was flagged by a human reviewer that was *not* also flagged by at least one LLM measure — humans are a strict subset of LLM detections
- The LLM-as-judge methods are more sensitive to minor unsupported claims, while human reviewers focus on claims that materially affect the response's usefulness

### 13.6 Best and Worst Items

**10 worst items** (lowest Claude composite score, averaging accuracy + completeness + helpfulness):

| ID | Composite | Acc | Comp | Help | Hall | Faith | ROUGE-1 | Question (truncated) |
|---|---|---|---|---|---|---|---|---|
| 13 | 1.00 | 1 | 1 | 1 | N | 5 | 0.15 | What is the vision behind Upstream? |
| 20 | 1.00 | 1 | 1 | 1 | N | 1 | 0.15 | How do I fund my Upstream Account using a Bank Wir… |
| 112 | 1.00 | 1 | 1 | 1 | Y | 1 | 0.26 | What is impermanent loss? |
| 137 | 1.00 | 1 | 1 | 1 | N | 5 | 0.14 | What might Upstream consider objectionable Collect… |
| 145 | 1.00 | 1 | 1 | 1 | Y | 1 | 0.27 | Where is Upstream located? |
| 146 | 1.00 | 1 | 1 | 1 | N | 2 | 0.54 | What annual & semi-annual disclosures are required… |
| 153 | 1.00 | 1 | 1 | 1 | Y | 2 | 0.41 | What is Short Selling on Upstream? |
| 116 | 1.33 | 2 | 1 | 1 | Y | 2 | 0.09 | What are Upstream's unique Collectible integration… |
| 126 | 1.33 | 2 | 1 | 1 | N | 2 | 0.19 | Can I cancel an auction before the close? |
| 162 | 1.33 | 1 | 2 | 1 | Y | 1 | 0.58 | What is Upstream's Alternative Uptick Rule? |

The worst items fall into two categories:
1. **Abstract/conceptual questions** (vision, impermanent loss) where the 2B model lacks the capacity for nuanced explanation
2. **Multi-step procedural questions** (funding via bank wire, collectible integrations) where crucial steps are omitted or fabricated

Interestingly, two items score 1/1/1 on quality dimensions yet 5 on faithfulness (items 13 and 137). These represent cases where the chatbot's response is faithful to *some* aspect of the source but entirely fails to address the user's actual question — highlighting that faithfulness and usefulness are fundamentally different constructs.

**10 best items** (highest Claude composite):

| ID | Composite | Acc | Comp | Help | Hall | Faith | ROUGE-1 | Question (truncated) |
|---|---|---|---|---|---|---|---|---|
| 67 | 5.00 | 5 | 5 | 5 | Y | 2 | 0.27 | What are the fees associated with trading? |
| 72 | 5.00 | 5 | 5 | 5 | N | 5 | 0.64 | How do I receive an application to list my company… |
| 117 | 5.00 | 5 | 5 | 5 | N | 5 | 0.54 | How do I Claim a Collectible? |
| 118 | 5.00 | 5 | 5 | 5 | N | 5 | 0.46 | How can I purchase a Collectible? |
| 122 | 5.00 | 5 | 5 | 5 | N | 5 | 0.70 | How can I sell a Collectible? |
| 125 | 5.00 | 5 | 5 | 5 | N | 5 | 0.48 | What happens if nobody bids at or above my reserve… |
| 130 | 5.00 | 5 | 5 | 5 | N | 4 | 0.45 | Where can I read the full Terms and Conditions for… |
| 151 | 5.00 | 5 | 5 | 5 | N | 5 | 0.62 | Where can I download the Upstream App? |
| 156 | 5.00 | 5 | 5 | 5 | N | 4 | 0.57 | Can a stock lender recall their shares early on Up… |
| 159 | 5.00 | 5 | 5 | 5 | N | 5 | 0.63 | What is the Downside Risk and Auto-Liquidation pro… |

The best items are predominantly **concrete how-to questions** with moderate-length ground truths. The model excels at reproducing procedural, fact-based answers where the expected response is well-defined and not overly long.

Item 67 is notable: it scores 5/5/5 on quality but is flagged for hallucination with a faithfulness score of only 2 — further illustrating the independence of quality and hallucination dimensions.

### 13.7 Metric Intercorrelation Matrix

Full Spearman correlation matrix across all measures (n = 164, * = p < 0.05):

|  | R1 | R2 | RL | BF1 | Acc | Cmp | Hlp | Fth | Lat | RLen | GLen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **R1** | 1.00 | | | | | | | | | | |
| **R2** | 0.89* | 1.00 | | | | | | | | | |
| **RL** | 0.93* | 0.95* | 1.00 | | | | | | | | |
| **BF1** | 0.80* | 0.86* | 0.84* | 1.00 | | | | | | | |
| **Acc** | 0.22* | 0.30* | 0.31* | 0.26* | 1.00 | | | | | | |
| **Cmp** | 0.43* | 0.48* | 0.47* | 0.51* | 0.53* | 1.00 | | | | | |
| **Hlp** | 0.44* | 0.44* | 0.47* | 0.47* | 0.69* | 0.82* | 1.00 | | | | |
| **Fth** | 0.30* | 0.30* | 0.33* | 0.26* | 0.74* | 0.34* | 0.55* | 1.00 | | | |
| **Lat** | 0.03 | 0.01 | −0.05 | 0.08 | −0.10 | −0.06 | −0.03 | −0.11 | 1.00 | | |
| **RLen** | 0.16* | 0.11 | 0.05 | 0.17* | −0.13 | 0.03 | 0.04 | −0.18* | 0.87* | 1.00 | |
| **GLen** | −0.04 | −0.13 | −0.17* | −0.15 | −0.11 | −0.41* | −0.21* | 0.13 | 0.60* | 0.47* | 1.00 |

**Legend**: R1 = ROUGE-1, R2 = ROUGE-2, RL = ROUGE-L, BF1 = BERTScore F₁, Acc = Claude Accuracy, Cmp = Claude Completeness, Hlp = Claude Helpfulness, Fth = Faithfulness, Lat = Latency, RLen = Response Word Count, GLen = Ground-Truth Word Count.

**Notable correlations**:
- **Accuracy ↔ Faithfulness (ρ = 0.74)**: The strongest cross-method correlation, indicating accurate responses are almost always faithful and vice versa
- **Helpfulness ↔ Completeness (ρ = 0.82)**: Helpfulness is heavily driven by completeness — incomplete answers are rarely helpful
- **Latency ↔ Response Length (ρ = 0.87)**: Longer responses take proportionally longer to generate, confirming the relationship is mechanical (token generation time) rather than "thinking time"
- **Ground-Truth Length ↔ Completeness (ρ = −0.41)**: The strongest negative correlation in the matrix — longer source answers systematically produce less complete responses
- **Latency ↔ All Quality Measures (ρ ≈ 0)**: Latency is completely orthogonal to quality

### 13.8 Largest Human-Claude Disagreements

The 10 items with the largest accuracy disagreement between human reviewers and Claude (human score minus Claude score):

| ID | Human Accuracy | Claude Accuracy | Δ | Question (truncated) |
|---|---|---|---|---|
| 145 | 5 | 1 | +4 | Where is Upstream located? |
| 153 | 5 | 1 | +4 | What is Short Selling on Upstream? |
| 162 | 5 | 1 | +4 | What is Upstream's Alternative Uptick Rule? |
| 146 | 5 | 1 | +4 | What annual & semi-annual disclosures are required? |
| 116 | 5 | 2 | +3 | What are Upstream's unique Collectible integrations? |
| 25 | 5 | 2 | +3 | Why Can't I Transfer USDC Directly from my Personal Wal… |
| 56 | 5 | 2 | +3 | How do I transfer shares to Upstream? |
| 36 | 5 | 2 | +3 | Can I open a Joint or Corporate account? |
| 112 | 4 | 1 | +3 | What is impermanent loss? |
| 106 | 5 | 2 | +3 | How can I withdraw my Market Pool earnings? |

In every case, the human rated the response as highly accurate (4–5) while Claude rated it as inaccurate (1–2). Manual inspection of these items reveals that the chatbot's responses typically provide a reasonable high-level answer that a domain expert would consider "good enough" but omit specific details or include minor inaccuracies that Claude's literal rubric application penalizes. This exemplifies the difference between domain-expert pragmatic evaluation and LLM strict-criteria evaluation.

---

## 14. Discussion and Key Findings

### 14.1 Finding 1: Automated Metrics Validate Completeness, Not Accuracy

ROUGE and BERTScore correlate moderately with completeness (ρ up to 0.65) but poorly with accuracy (ρ near zero for humans). This means automated metrics can reliably estimate *how much* of the reference content a response covers, but cannot determine *whether the content is correct*. Evaluation pipelines that rely solely on ROUGE/BERTScore will miss factual errors and hallucinations entirely.

**Implication**: ROUGE/BERTScore should be used as a completeness proxy, not a quality proxy. Accuracy assessment requires either LLM-as-judge or human evaluation.

### 14.2 Finding 2: Ground-Truth Complexity is the Primary Quality Driver

The chatbot's weakest point is completeness on complex, multi-part answers (ρ = −0.41 between ground-truth length and completeness). This is likely a fundamental capacity limitation of the 2B parameter model: with only ~1.5 GB of quantized weights, the model cannot reliably maintain and reproduce all key points from a long source passage.

**Implication**: For deployment, consider implementing answer segmentation — breaking complex FAQ entries into multiple sub-questions with shorter answers — to keep within the model's effective context utilization window.

### 14.3 Finding 3: LLM-as-Judge Detects Issues Humans Miss

33% of items were flagged by both LLM-based methods (faithfulness scoring and Claude review) but not by human reviewers. This indicates LLM evaluators apply a more granular definition of "hallucination," catching minor unsupported claims that domain experts overlook or consider immaterial.

**Implication**: LLM-as-judge evaluation complements rather than replaces human review. The two approaches have different sensitivity profiles: LLMs catch fine-grained factual deviations; humans assess practical impact.

### 14.4 Finding 4: Hallucination is an Addendum, Not a Disqualifier

19.5% of items are flagged for hallucination yet score ≥ 4 on accuracy. The hallucination flag means "at least one unsupported claim is present," not "the response is unreliable." In many cases, the unsupported claim is a minor detail (e.g., a slightly extrapolated process step) embedded in an otherwise accurate and helpful response.

Only 12.8% of items are both hallucinated and substantially inaccurate (accuracy ≤ 2). These represent the genuinely problematic responses requiring intervention.

**Implication**: Hallucination rates in isolation are misleading quality indicators. They should always be reported alongside accuracy scores to contextualize severity.

### 14.5 Finding 5: Claude Exhibits Systematic Harshness Bias

Claude scores approximately 1 point lower than human reviewers across all dimensions and flags hallucinations 7–22× more frequently. This is directional bias, not random error — Claude and humans rank items similarly but apply different absolute thresholds.

**Implication**: When using LLM-as-judge scores for benchmarking across studies, raw scores are not directly comparable to human annotations. Either calibrate using paired human-LLM data or report both.

### 14.6 Finding 6: Latency is Orthogonal to Quality

No significant correlation exists between inference latency and any quality metric. Faster responses are not worse, and slower responses are not better. The latency variation (6–61 seconds) is driven primarily by response length (ρ = 0.87), which is a mechanical relationship (more tokens take longer to generate) rather than a quality-related one.

**Implication**: Latency optimization efforts can focus purely on throughput without concern for quality degradation. The model's quality is determined at prompt construction and retrieval time, not during generation.

---

## 15. Limitations and Threats to Validity

### 15.1 Internal Validity

- **Single evaluation run**: All chatbot responses were collected in a single session (March 23, 2026). Model behavior may vary across restarts due to the stochastic nature of LLM generation. Repeated evaluations would strengthen confidence in the results.
- **UAT endpoint**: The evaluation used the UAT environment rather than production. While the model and knowledge base are identical, any environment-specific configuration differences (e.g., memory allocation, concurrent load) could affect results.
- **LLM judge model version**: Claude Sonnet 4 was the evaluator. Different model versions, or different judge models (e.g., GPT-4, Gemini), may produce different scores. The systematic harshness bias observed may be specific to this model family.

### 15.2 External Validity

- **Single system**: All findings are based on one chatbot system (Gemma 2B + FAISS RAG on Azure App Service). Generalization to other model sizes, quantization levels, retrieval methods, or deployment configurations requires additional evaluation.
- **Domain specificity**: The knowledge base covers financial trading platform FAQs. Results may not generalize to other FAQ domains with different question types, answer complexity profiles, or terminology.

### 15.3 Construct Validity

- **Small expert pool**: Only two human reviewers evaluated 50 items each, with 15 overlapping. Inter-rater agreement statistics based on 15 items have wide confidence intervals. A larger reviewer pool and sample size would improve reliability estimates.
- **Rubric interpretation**: Despite using the same rubric, human reviewers and Claude applied different thresholds (§12.4). The rubric may benefit from additional calibration examples or more specific anchoring descriptions.
- **Hallucination definition**: The "at least one unsupported claim" threshold for the binary hallucination flag produces high flag rates that may overstate the practical impact of hallucination. A severity-weighted definition might be more informative.

### 15.4 Conclusion Validity

- **Ceiling effect**: Human reviewers' scores cluster at 4–5, limiting the discriminative power of human evaluation and compressing the range for agreement statistics. This is a known limitation of Likert scales when applied to relatively high-quality systems.
- **Correlation vs. causation**: The correlation analyses in §13 identify statistical associations but do not establish causal relationships. For example, the negative correlation between ground-truth length and completeness could reflect model capacity limitations, retrieval quality degradation on longer passages, or prompt construction effects.

---

## 16. Reproducibility Guide

### 16.1 File Inventory

```
evaluation/
├── EVALUATION_REPORT.md          ← This document
├── setup_env.ps1                 ← Environment setup (API key prompt)
├── analyze_hallucinations.py     ← Hallucination type categorization
├── cross_tab.py                  ← Hallucination × accuracy cross-tabulation
│
├── data/
│   ├── build_dataset.py          ← Dataset construction script
│   ├── dataset.json              ← Collected dataset (164 items)
│   └── requirements.txt          ← Data collection dependencies
│
├── automated-metrics/
│   ├── compute_metrics.py        ← ROUGE + BERTScore computation
│   ├── compute_faithfulness.py   ← Claude faithfulness scoring
│   ├── compute_claude_review.py  ← Claude multi-dimensional review
│   ├── compute_agreement.py      ← Inter-rater agreement statistics
│   ├── deep_analysis.py          ← Correlation and pattern analyses
│   └── requirements.txt          ← Analysis dependencies
│
├── expert-review/
│   ├── index.html                ← Expert review SPA
│   ├── app.js                    ← Review interface logic
│   └── style.css                 ← Review interface styling
│
└── results/
    ├── automated_metrics.json         ← ROUGE + BERTScore results
    ├── faithfulness_scores.json       ← Faithfulness scoring results
    ├── claude_review_scores.json      ← Multi-dimensional review results
    ├── evaluation-Andy-2026-03-23.json        ← Reviewer A data
    └── evaluation-TimLovett-2026-03-24.json   ← Reviewer B data
```

### 16.2 Environment Setup

**Python version**: 3.12.5 (CPython)  
**Operating system**: Windows 10/11  

```powershell
# Install dependencies
pip install requests rouge-score bert-score torch anthropic scikit-learn krippendorff scipy numpy

# Set API key (required for faithfulness and Claude review scripts)
$env:ANTHROPIC_API_KEY = "your-key-here"
```

### 16.3 Execution Order

Scripts should be run in this order (each depends on outputs of prior steps):

1. **Build dataset**: `python evaluation/data/build_dataset.py`  
   Requires: live chatbot endpoint. Produces: `dataset.json`

2. **Compute automated metrics**: `python evaluation/automated-metrics/compute_metrics.py`  
   Requires: `dataset.json`. Produces: `automated_metrics.json`

3. **Compute faithfulness**: `python evaluation/automated-metrics/compute_faithfulness.py`  
   Requires: `dataset.json`, `ANTHROPIC_API_KEY`. Produces: `faithfulness_scores.json`

4. **Compute Claude review**: `python evaluation/automated-metrics/compute_claude_review.py`  
   Requires: `dataset.json`, `ANTHROPIC_API_KEY`. Produces: `claude_review_scores.json`

5. **Collect expert reviews**: Serve `evaluation/expert-review/` via HTTP server, have reviewers complete reviews and export JSON files to `results/`

6. **Compute agreement**: `python evaluation/automated-metrics/compute_agreement.py`  
   Requires: `claude_review_scores.json`, `evaluation-*.json` files

7. **Run deep analysis**: `python evaluation/automated-metrics/deep_analysis.py`  
   Requires: all result files

8. **Analyze hallucinations**: `python evaluation/analyze_hallucinations.py`  
   Requires: `faithfulness_scores.json`

9. **Cross-tabulate**: `python evaluation/cross_tab.py`  
   Requires: `claude_review_scores.json`

### 16.4 Expected Runtimes

| Step | Approximate Duration | Notes |
|---|---|---|
| Dataset construction | ~45 minutes | 164 queries × ~14s each + 0.5s delays |
| Automated metrics | ~5 minutes | BERTScore model download on first run |
| Faithfulness scoring | ~15–20 minutes | 163 API calls to Claude |
| Claude review | ~15–20 minutes | 164 API calls to Claude |
| Expert reviews | ~2–3 hours each | Manual human effort |
| Agreement + deep analysis | < 1 minute | Local computation only |

### 16.5 API Costs

The evaluation requires approximately 330 API calls to Claude Sonnet 4 (163 for faithfulness + 164 for multi-dimensional review + retries). At typical API pricing, this represents a modest cost. Exact amounts depend on current Anthropic pricing and the volume of input/output tokens per call.

---

## 17. References

- Chiang, C.-H., & Lee, H. (2023). Can large language models be an alternative to human evaluations? *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)*.

- Cohen, J. (1968). Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit. *Psychological Bulletin*, 70(4), 213–220.

- Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y., Madotto, A., & Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12), 1–38.

- Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability. *Departmental Papers (ASC)*, University of Pennsylvania.

- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159–174.

- Likert, R. (1932). A technique for the measurement of attitudes. *Archives of Psychology*, 22(140), 1–55.

- Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text Summarization Branches Out: Proceedings of the ACL-04 Workshop*, 74–81.

- Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On faithfulness and factuality in abstractive summarization. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 1906–1919.

- van der Lee, C., Gatt, A., van Miltenburg, E., & Krahmer, E. (2021). Human evaluation of automatically generated text: A survey. *Journal of Artificial Intelligence Research*, 72, 801–840.

- Wohlin, C., Runeson, P., Höst, M., Ohlsson, M. C., Regnell, B., & Wesslén, A. (2012). *Experimentation in Software Engineering*. Springer.

- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. *Proceedings of the Eighth International Conference on Learning Representations (ICLR)*.

- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.
