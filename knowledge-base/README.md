# Knowledge Base Description

## Overview

The domain knowledge base used in this study comprises **164 question–answer pairs** for the Upstream digital securities marketplace ([https://upstream.exchange](https://upstream.exchange)).

## Categories

The Q&A pairs span the following topic categories:

- **Account Management** — Account creation, verification, settings, password recovery
- **Trading Operations** — Order types, trading hours, fees, execution
- **Collectibles** — Digital collectible items, purchasing, ownership
- **Platform Features** — App functionality, notifications, referrals
- **Regulatory Topics** — Compliance, jurisdiction, investor protection

## Answer Complexity

Answer complexity varies across the corpus:

- **Simple responses** — Single-sentence factual answers (e.g., "What are the trading hours?")
- **Moderate responses** — Multi-sentence explanations with procedural steps
- **Complex responses** — Multi-paragraph explanations covering multiple aspects or edge cases

The quality evaluation in the paper found that answer complexity is the primary driver of quality degradation in the 2B-parameter model's output.

## Data Availability

The Q&A content is proprietary to Upstream and cannot be redistributed. Researchers wishing to replicate the evaluation methodology should prepare a comparable domain-specific Q&A corpus with a similar distribution of answer complexities.

## Embedding and Indexing

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Vector index**: FAISS (Facebook AI Similarity Search)
- **Retrieval parameter**: top-k = 2
