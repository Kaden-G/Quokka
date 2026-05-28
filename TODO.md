# Quokka eval harness — cleanup backlog

Captured from a comment pass on `evaluate.py`. None are urgent; parked here so
they're out of working memory. Backburner unless this becomes active portfolio polish.

## 1. `mkdir` won't create parent dirs
- **Where:** `QuokkaEvaluator.__init__`
- **Issue:** `self.results_dir.mkdir(exist_ok=True)` throws `FileNotFoundError` if `data/` doesn't exist yet.
- **Fix:** `self.results_dir.mkdir(parents=True, exist_ok=True)`
- **Effort:** trivial (1 word)
- **Ref:** https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir

## 2. Cache-hit detection is timing-based / noisy
- **Where:** `benchmark_performance`
- **Issue:** `cached_latency < latency / 10` infers a cache hit from timing — a GC pause or busy CPU can flip it.
- **Fix:** expose a real hit/miss counter on `SOPSearcher.query_cache` and read that instead.
- **Effort:** small–medium (depends on touching SOPSearcher)

## 3. Benchmark queries are degenerate (timing only, no quality signal)
- **Where:** `benchmark_performance`
- **Issue:** uses chunk text as its own query, so every query trivially matches its source. Fine for latency, meaningless for ranking quality.
- **Fix:** keep as-is for speed; rely on `evaluate_test_set` for quality. (Already documented in the docstring — no code change needed, just don't mistake the numbers.)
- **Effort:** none — awareness only

## 4. Harness reaches into searcher internals
- **Where:** `compare_configurations` (and the cache check in #2)
- **Issue:** `self.searcher.query_cache.clear()` couples the harness to SOPSearcher's internals; breaks if that attribute is renamed.
- **Fix:** add a public `SOPSearcher.clear_cache()` and call that.
- **Effort:** small

## 5. Minor cleanups
- `Tuple` imported but never used → drop it.
- `evaluate_test_set` indexes `all_metrics[0]` → `IndexError` on an empty test set; guard upstream.
- **Effort:** trivial
