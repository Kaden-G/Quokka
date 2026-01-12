#!/usr/bin/env python3
"""
Quokka Evaluation Suite
Measures search quality and system performance.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
from search import SOPSearcher


class QuokkaEvaluator:
    """Evaluate Quokka search performance."""

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.searcher = SOPSearcher(str(index_dir))
        self.results_dir = Path(__file__).parent.parent / 'data' / 'evaluation'
        self.results_dir.mkdir(exist_ok=True)

    def evaluate_query(
        self,
        query: str,
        relevant_chunks: List[str],
        top_k: int = 5
    ) -> Dict:
        """
        Evaluate a single query.

        Args:
            query: Search query
            relevant_chunks: List of chunk_ids that are relevant
            top_k: Number of results to evaluate

        Returns:
            Metrics dictionary
        """
        # Time the search
        start = time.time()
        results = self.searcher.search(query, top_k=top_k, rerank=True)
        latency = time.time() - start

        # Extract retrieved chunk IDs
        retrieved = [r['chunk_id'] for r in results]

        # Calculate metrics
        metrics = {
            'query': query,
            'latency_ms': latency * 1000,
            'num_results': len(results),
            'relevant_chunks': relevant_chunks,
            'retrieved_chunks': retrieved,
        }

        # Precision@K
        relevant_retrieved = set(retrieved[:top_k]) & set(relevant_chunks)
        metrics['precision@k'] = len(relevant_retrieved) / top_k if top_k > 0 else 0

        # Recall@K
        metrics['recall@k'] = len(relevant_retrieved) / len(relevant_chunks) if relevant_chunks else 0

        # F1@K
        if metrics['precision@k'] + metrics['recall@k'] > 0:
            metrics['f1@k'] = 2 * (metrics['precision@k'] * metrics['recall@k']) / \
                              (metrics['precision@k'] + metrics['recall@k'])
        else:
            metrics['f1@k'] = 0

        # Mean Reciprocal Rank (MRR)
        for i, chunk_id in enumerate(retrieved):
            if chunk_id in relevant_chunks:
                metrics['mrr'] = 1.0 / (i + 1)
                break
        else:
            metrics['mrr'] = 0.0

        # Normalized Discounted Cumulative Gain (NDCG)
        metrics['ndcg@k'] = self._calculate_ndcg(retrieved, relevant_chunks, top_k)

        # Average scores
        if results:
            metrics['avg_similarity'] = np.mean([r['similarity'] for r in results])
            if 'rerank_score' in results[0]:
                metrics['avg_rerank_score'] = np.mean([r['rerank_score'] for r in results])

        return metrics

    def _calculate_ndcg(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        # DCG
        dcg = 0.0
        for i, chunk_id in enumerate(retrieved[:k]):
            if chunk_id in relevant:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0

        # IDCG (ideal DCG if all relevant items were at top)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_test_set(self, test_queries: List[Dict]) -> Dict:
        """
        Evaluate multiple queries.

        Args:
            test_queries: List of {"query": str, "relevant": [chunk_ids]}

        Returns:
            Aggregated metrics
        """
        all_metrics = []

        print(f"\nEvaluating {len(test_queries)} test queries...")
        for i, test in enumerate(test_queries, 1):
            print(f"  [{i}/{len(test_queries)}] {test['query'][:50]}...")
            metrics = self.evaluate_query(
                test['query'],
                test['relevant'],
                top_k=5
            )
            all_metrics.append(metrics)

        # Aggregate
        aggregated = {
            'num_queries': len(test_queries),
            'avg_precision@5': np.mean([m['precision@k'] for m in all_metrics]),
            'avg_recall@5': np.mean([m['recall@k'] for m in all_metrics]),
            'avg_f1@5': np.mean([m['f1@k'] for m in all_metrics]),
            'avg_mrr': np.mean([m['mrr'] for m in all_metrics]),
            'avg_ndcg@5': np.mean([m['ndcg@k'] for m in all_metrics]),
            'avg_latency_ms': np.mean([m['latency_ms'] for m in all_metrics]),
            'p95_latency_ms': np.percentile([m['latency_ms'] for m in all_metrics], 95),
        }

        if 'avg_similarity' in all_metrics[0]:
            aggregated['avg_similarity'] = np.mean([m['avg_similarity'] for m in all_metrics])
        if 'avg_rerank_score' in all_metrics[0]:
            aggregated['avg_rerank_score'] = np.mean([m['avg_rerank_score'] for m in all_metrics])

        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'eval_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump({
                'aggregated': aggregated,
                'individual_queries': all_metrics,
                'timestamp': timestamp
            }, f, indent=2)

        print(f"\nResults saved to: {results_file}")

        return aggregated

    def benchmark_performance(self, num_queries: int = 100) -> Dict:
        """Benchmark system performance with random queries."""
        # Generate test queries from indexed chunks
        sample_chunks = np.random.choice(
            self.searcher.metadata,
            size=min(num_queries, len(self.searcher.metadata)),
            replace=False
        )

        latencies = []
        cache_hits = 0

        print(f"\nBenchmarking with {len(sample_chunks)} queries...")
        for i, chunk in enumerate(sample_chunks, 1):
            # Use chunk text as query
            query = chunk['text'][:100]  # First 100 chars

            # Measure latency
            start = time.time()
            results = self.searcher.search(query, top_k=5)
            latency = time.time() - start
            latencies.append(latency * 1000)

            # Test cache (second query should be faster)
            start = time.time()
            _ = self.searcher.search(query, top_k=5)
            cached_latency = time.time() - start
            if cached_latency < latency / 10:  # 10x faster = cache hit
                cache_hits += 1

            if i % 20 == 0:
                print(f"  Progress: {i}/{len(sample_chunks)}")

        return {
            'num_queries': len(latencies),
            'avg_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'cache_hit_rate': cache_hits / len(latencies),
        }

    def compare_configurations(self) -> Dict:
        """Compare different search configurations."""
        test_query = "What are the safety procedures?"

        configs = {
            'baseline': {'rerank': False},
            'with_reranking': {'rerank': True},
        }

        results = {}

        print("\nComparing configurations...")
        for name, config in configs.items():
            print(f"  Testing: {name}")

            # Measure latency
            latencies = []
            for _ in range(10):
                start = time.time()
                search_results = self.searcher.search(test_query, top_k=5, **config)
                latency = time.time() - start
                latencies.append(latency * 1000)

            results[name] = {
                'avg_latency_ms': np.mean(latencies),
                'top_similarity': search_results[0]['similarity'] if search_results else 0,
                'top_rerank_score': search_results[0].get('rerank_score', 0) if search_results else 0,
            }

        return results


def create_sample_test_set() -> List[Dict]:
    """
    Create a sample test set for evaluation.

    Users should replace this with real labeled data.
    """
    return [
        {
            'query': 'What safety equipment is required?',
            'relevant': []  # User should fill in relevant chunk IDs
        },
        {
            'query': 'How to initialize the system?',
            'relevant': []
        },
        {
            'query': 'Emergency shutdown procedure',
            'relevant': []
        },
    ]


def main():
    """Run evaluation suite."""
    base_dir = Path(__file__).parent.parent
    index_dir = base_dir / 'data' / 'index'

    evaluator = QuokkaEvaluator(str(index_dir))

    print("="*80)
    print("Quokka Evaluation Suite")
    print("="*80)

    # 1. Performance Benchmark
    print("\n[1] Performance Benchmark")
    print("-" * 80)
    perf_results = evaluator.benchmark_performance(num_queries=50)
    print("\nPerformance Results:")
    for metric, value in perf_results.items():
        if 'latency' in metric:
            print(f"  {metric}: {value:.2f} ms")
        elif 'rate' in metric:
            print(f"  {metric}: {value:.1%}")
        else:
            print(f"  {metric}: {value}")

    # 2. Configuration Comparison
    print("\n[2] Configuration Comparison")
    print("-" * 80)
    config_results = evaluator.compare_configurations()
    print("\nConfiguration Results:")
    for config, metrics in config_results.items():
        print(f"\n  {config}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")

    # 3. Test Set Evaluation (if available)
    print("\n[3] Test Set Evaluation")
    print("-" * 80)
    print("\nTo run test set evaluation:")
    print("  1. Create labeled test queries in data/evaluation/test_queries.json")
    print("  2. Format: [{'query': '...', 'relevant': ['chunk_id1', ...]}]")
    print("  3. Run: evaluator.evaluate_test_set(test_queries)")

    test_file = base_dir / 'data' / 'evaluation' / 'test_queries.json'
    if test_file.exists():
        with open(test_file) as f:
            test_queries = json.load(f)
        eval_results = evaluator.evaluate_test_set(test_queries)
        print("\nEvaluation Results:")
        for metric, value in eval_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    else:
        print(f"\nNo test file found at: {test_file}")
        print("Using default evaluation metrics from benchmarks above.")

    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == '__main__':
    main()
