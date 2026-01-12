# Quokka Metrics & Evaluation Guide

**How to Measure Success in Your SOP Search System**

---

## Table of Contents

1. [Overview](#overview)
2. [Key Metrics](#key-metrics)
3. [Evaluation Tools](#evaluation-tools)
4. [Setting Up Metrics Tracking](#setting-up-metrics-tracking)
5. [Creating Test Sets](#creating-test-sets)
6. [Running Evaluations](#running-evaluations)
7. [Interpreting Results](#interpreting-results)
8. [Continuous Improvement](#continuous-improvement)

---

## Overview

Measuring search system performance requires both **automated metrics** (precision, recall, latency) and **user feedback** (relevance ratings, satisfaction).

### What We Track

**Automated Metrics:**
- Search quality (precision, recall, NDCG)
- Performance (latency, throughput)
- System health (cache hit rate, error rate)

**User Feedback:**
- Relevance ratings (1-5 stars)
- Which results were helpful
- Comments and suggestions

---

## Key Metrics

### 1. Search Quality Metrics

| Metric | Description | Target | How to Improve |
|--------|-------------|--------|----------------|
| **Precision@5** | % of top 5 results that are relevant | >70% | Enable re-ranking, improve embeddings |
| **Recall@5** | % of all relevant docs in top 5 | >60% | Increase retrieval candidates, better chunking |
| **NDCG@5** | Normalized quality (considers ranking) | >0.75 | Re-ranking, better similarity metric |
| **MRR** | Mean Reciprocal Rank (1/rank of first relevant) | >0.70 | Re-ranking, query expansion |
| **Avg Similarity** | Average cosine similarity score | >0.65 | Better embeddings (bge-small), more data |

### 2. Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Latency (p50)** | Median query time | <300ms |
| **Latency (p95)** | 95th percentile | <800ms |
| **Latency (p99)** | 99th percentile | <2000ms |
| **Cache Hit Rate** | % of queries served from cache | >20% |
| **Throughput** | Queries per second | >10 QPS |

### 3. User Satisfaction Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Avg Rating** | Average 1-5 star rating | >4.0 |
| **Satisfaction Rate** | % of 4-5 star ratings | >75% |
| **Feedback Rate** | % of queries with feedback | >10% |

---

## Evaluation Tools

### 1. Automated Evaluation (`scripts/evaluate.py`)

**Purpose:** Measure search quality on labeled test set

**Features:**
- Precision@K, Recall@K, F1@K
- NDCG@K (ranking quality)
- MRR (first relevant rank)
- Latency benchmarks
- Configuration comparisons

**Usage:**
```bash
python scripts/evaluate.py
```

**Output:**
```
Performance Benchmark:
  avg_latency_ms: 287.45
  p95_latency_ms: 542.12
  cache_hit_rate: 24.5%

Configuration Comparison:
  baseline:
    avg_latency_ms: 145.23
    top_similarity: 0.6234
  with_reranking:
    avg_latency_ms: 398.67
    top_similarity: 0.7123
    top_rerank_score: 0.9012
```

### 2. Metrics Tracking (`scripts/metrics.py`)

**Purpose:** Log all queries and collect user feedback over time

**Features:**
- SQLite database for historical data
- Query statistics (latency, similarity, cache hits)
- User feedback collection
- Top queries analysis
- Metrics export (JSON)

**Usage:**
```bash
# View current metrics
python scripts/metrics.py

# In your application
from metrics import MetricsTracker
tracker = MetricsTracker()

# Log query
query_id = tracker.log_query(query, results, latency)

# Collect feedback
tracker.log_feedback(query_id, rating=4, relevant_results=['chunk_id_1'])

# Get stats
stats = tracker.get_query_stats(days=7)
```

### 3. Live Metrics API

**Endpoints:**

**Get Metrics:**
```bash
curl http://127.0.0.1:5000/api/metrics?days=7
```

**Submit Feedback:**
```bash
curl -X POST http://127.0.0.1:5000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": 123,
    "rating": 5,
    "relevant_results": ["chunk_id_1", "chunk_id_3"],
    "comments": "Very helpful!"
  }'
```

---

## Setting Up Metrics Tracking

### 1. Enable Metrics in Server

Metrics tracking is automatically enabled when you start the server:

```bash
python app/server.py
```

Output:
```
Search engine initialized with 236 chunks
Metrics tracking enabled
```

### 2. Database Location

Metrics are stored in: `data/metrics/quokka_metrics.db`

**Tables:**
- `queries` - All search queries and results
- `feedback` - User ratings and comments
- `system_performance` - System metrics

### 3. Viewing Metrics

```bash
# Command line
python scripts/metrics.py

# API
curl http://127.0.0.1:5000/api/metrics

# Export to JSON
python -c "from scripts.metrics import MetricsTracker; \
  MetricsTracker().export_metrics('metrics.json')"
```

---

## Creating Test Sets

### Why You Need Test Sets

- Objective measurement of improvements
- Compare different configurations
- Track performance over time
- Identify weak areas

### Creating a Test Set

**Format:** `data/evaluation/test_queries.json`

```json
[
  {
    "query": "What safety equipment is required for hazardous materials?",
    "relevant": [
      "GSE_SOP_NASA_p20_c1",
      "GSE_SOP_NASA_p85_c1"
    ]
  },
  {
    "query": "How do I initialize the cooling system?",
    "relevant": [
      "GSE_SOP_NASA_p42_c2"
    ]
  }
]
```

### How to Find Relevant Chunks

**Method 1: Manual Labeling**
```bash
# Search for query
python scripts/search.py

# Enter query: "safety equipment"
# Review results, note chunk_ids that are relevant
# Add to test_queries.json
```

**Method 2: Expert Review**
- Ask domain experts to review search results
- Have them mark which results answer the question
- Collect chunk_ids

**Method 3: Click Tracking**
- Track which results users click on
- Assume clicked results are relevant
- Build test set from real usage data

### Example Test Set Creation

```python
# scripts/create_test_set.py
from search import SOPSearcher

searcher = SOPSearcher('data/index')

test_queries = [
    "What safety equipment is required?",
    "How to shut down in emergency?",
    "Calibration procedures for sensors",
]

test_set = []
for query in test_queries:
    print(f"\nQuery: {query}")
    results = searcher.search(query, top_k=10)

    # Show results
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['chunk_id']}: {r['text'][:100]}...")

    # Get user input
    relevant = input("Enter relevant chunk numbers (comma-separated): ")
    relevant_ids = [results[int(i)-1]['chunk_id'] for i in relevant.split(',')]

    test_set.append({
        'query': query,
        'relevant': relevant_ids
    })

# Save
import json
with open('data/evaluation/test_queries.json', 'w') as f:
    json.dump(test_set, f, indent=2)
```

---

## Running Evaluations

### 1. Quick Performance Check

```bash
python scripts/evaluate.py
```

**What it tests:**
- Latency benchmarks (50 random queries)
- Configuration comparison (with/without re-ranking)
- System health

**When to run:** After making changes, weekly monitoring

### 2. Full Evaluation (with Test Set)

```bash
# Create test set (one-time)
python scripts/create_test_set.py

# Run evaluation
python scripts/evaluate.py

# Check results
cat data/evaluation/eval_*.json
```

**What it measures:**
- Precision@5, Recall@5, F1@5
- NDCG@5 (ranking quality)
- MRR (mean reciprocal rank)
- Per-query metrics

**When to run:** Before/after major changes, monthly

### 3. A/B Testing

Compare different configurations:

```python
# scripts/ab_test.py
from evaluate import QuokkaEvaluator

evaluator = QuokkaEvaluator('data/index')

# Test A: Baseline
searcher_a = SOPSearcher('data/index', use_reranker=False)

# Test B: With re-ranking
searcher_b = SOPSearcher('data/index', use_reranker=True)

# Compare on test set
results_a = evaluator.evaluate_test_set(test_queries, searcher_a)
results_b = evaluator.evaluate_test_set(test_queries, searcher_b)

print(f"Baseline NDCG: {results_a['avg_ndcg@5']:.3f}")
print(f"Re-ranked NDCG: {results_b['avg_ndcg@5']:.3f}")
print(f"Improvement: {(results_b['avg_ndcg@5'] - results_a['avg_ndcg@5']):.3f}")
```

---

## Interpreting Results

### Understanding Metrics

**Precision@5 = 0.80 (80%)**
- 4 out of 5 top results are relevant
- **Good:** Users find what they need
- **Improve:** Enable re-ranking

**Recall@5 = 0.50 (50%)**
- Only half of relevant docs in top 5
- **Issue:** Might be missing relevant docs
- **Improve:** Better embedding model, more retrieval candidates

**NDCG@5 = 0.75**
- Good ranking (1.0 is perfect)
- **Interpretation:** Relevant docs are usually in top positions
- **Improve:** Re-ranking helps here

**Latency p95 = 450ms**
- 95% of queries complete under 450ms
- **Good:** Users get fast results
- **Monitor:** Should stay under 1000ms

**Cache Hit Rate = 18%**
- 18% of queries served instantly from cache
- **Expected:** 15-30% for typical usage
- **Improve:** Longer cache retention

### What Good Looks Like

| Metric | Poor | Okay | Good | Excellent |
|--------|------|------|------|-----------|
| Precision@5 | <50% | 50-70% | 70-85% | >85% |
| NDCG@5 | <0.60 | 0.60-0.75 | 0.75-0.85 | >0.85 |
| Latency (p95) | >2s | 1-2s | 500ms-1s | <500ms |
| Avg Rating | <3.5 | 3.5-4.0 | 4.0-4.5 | >4.5 |
| Satisfaction | <60% | 60-75% | 75-85% | >85% |

### Red Flags

🚨 **Precision@5 < 50%**: Many irrelevant results - check embeddings, enable re-ranking

🚨 **Latency p95 > 2s**: Too slow - disable features, optimize index

🚨 **Avg Rating < 3.5**: Users unhappy - review feedback comments, improve results

🚨 **Cache Hit Rate < 5%**: Cache not working - check cache size, query diversity

---

## Continuous Improvement

### Monthly Review Process

**1. Collect Metrics (First week of month)**
```bash
# Export last month's data
python -c "from scripts.metrics import MetricsTracker; \
  MetricsTracker().export_metrics('metrics_$(date +%Y%m).json')"
```

**2. Analyze Trends**
- Compare to previous month
- Identify degradation or improvement
- Review user feedback comments

**3. Identify Issues**
- Queries with low ratings
- Slow queries (high latency)
- Common query patterns

**4. Prioritize Improvements**
- Fix critical issues first (low precision, high latency)
- Optimize common queries
- Implement feature requests

**5. Test Changes**
```bash
# Before change
python scripts/evaluate.py > before.txt

# Make improvements
# ...

# After change
python scripts/evaluate.py > after.txt

# Compare
diff before.txt after.txt
```

### Improvement Checklist

**Search Quality:**
- [ ] Precision@5 > 70%
- [ ] NDCG@5 > 0.75
- [ ] Avg similarity > 0.65
- [ ] Re-ranking enabled
- [ ] Using bge-small or better model

**Performance:**
- [ ] p95 latency < 800ms
- [ ] Cache hit rate > 15%
- [ ] No errors in logs
- [ ] Memory usage stable

**User Satisfaction:**
- [ ] Avg rating > 4.0
- [ ] Collecting feedback on >10% of queries
- [ ] Responding to user comments
- [ ] Iterating based on feedback

---

## Example: Monthly Metrics Report

```
=================================================================
Quokka Monthly Metrics Report - January 2026
=================================================================

USAGE STATISTICS (Last 30 days)
-----------------------------------------------------------------
Total Queries:               1,247
Unique Queries:                892
Avg Queries/Day:                42
Cache Hit Rate:              23.4%

SEARCH QUALITY
-----------------------------------------------------------------
Avg Precision@5:            0.78 (78%)  [Target: >70%] ✓
Avg NDCG@5:                 0.81        [Target: >0.75] ✓
Avg Similarity:             0.68        [Target: >0.65] ✓
Avg Re-rank Score:          0.84

PERFORMANCE
-----------------------------------------------------------------
Avg Latency:              312 ms
p95 Latency:              687 ms  [Target: <800ms] ✓
p99 Latency:            1,234 ms  [Target: <2000ms] ✓
Cache Hit Rate:          23.4%    [Target: >15%] ✓

USER SATISFACTION
-----------------------------------------------------------------
Total Feedback:                156  (12.5% of queries)
Avg Rating:                   4.2  [Target: >4.0] ✓
4-5 Star Ratings:            79%  [Target: >75%] ✓
1-2 Star Ratings:             8%

TOP QUERIES (by frequency)
-----------------------------------------------------------------
1. "safety equipment requirements" (43 queries, 4.5★)
2. "emergency shutdown procedure" (38 queries, 4.7★)
3. "calibration procedures" (31 queries, 4.1★)

AREAS FOR IMPROVEMENT
-----------------------------------------------------------------
- Low-rated queries: "troubleshooting sensor errors" (2.3★)
  → Action: Review relevant chunks, improve coverage
- High latency: Queries with "comprehensive" averaging 1.8s
  → Action: Optimize re-ranking for long queries

CHANGES THIS MONTH
-----------------------------------------------------------------
- Enabled cross-encoder re-ranking → +15% NDCG improvement
- Switched to bge-small embeddings → +12% precision improvement
- Added query caching → 23% hit rate

NEXT MONTH GOALS
-----------------------------------------------------------------
- Improve "troubleshooting" query results
- Reduce p99 latency to <1000ms
- Increase feedback rate to >15%
=================================================================
```

---

## Quick Reference

**Daily:**
```bash
# Check if system is healthy
curl http://127.0.0.1:5000/health
```

**Weekly:**
```bash
# Review metrics
python scripts/metrics.py

# Check for degradation
curl http://127.0.0.1:5000/api/metrics?days=7
```

**Monthly:**
```bash
# Full evaluation
python scripts/evaluate.py

# Export metrics report
python -c "from scripts.metrics import MetricsTracker; \
  MetricsTracker().export_metrics('monthly_report.json')"

# Review user feedback
sqlite3 data/metrics/quokka_metrics.db \
  "SELECT * FROM feedback WHERE rating <= 2;"
```

**After Changes:**
```bash
# A/B test
python scripts/evaluate.py > before.txt
# ... make changes ...
python scripts/evaluate.py > after.txt
diff before.txt after.txt
```

---

**Remember:** Metrics only matter if you act on them. Review regularly, iterate based on data, and keep improving!
