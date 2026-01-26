# Quokka Eval Harness (Phase 0)

This folder is a **locked experiment harness** for Quokka.

It gives you:
- `eval/questions.jsonl` — a starter golden query set (30–50 queries)
- A **single command** to run the pipeline over the query set
- A **per-query run record** saved to JSONL + human-reviewable artifacts
- Semi-manual scorers (precision@k proxy, completeness checklist, citation coverage)

The harness is intentionally **pipeline-agnostic**: today it runs a deterministic `mock` pipeline that only extracts from local documents. Later you’ll swap in the real LMCO embedding + retrieval + LMCO LLM providers without changing the harness contract.

## Quick start (Windows-friendly)

1) Put your documents under a local `data/` folder (relative to this repo):

```
quokka_eval/
  data/
    SOPs/
    SmartBooks/
    ...
```

2) Run the evaluation:

```powershell
python -m quokka_eval.run --config .\configs\baseline.toml
```

3) Review artifacts under `runs/<run_id>/`:
- `run.jsonl` (machine-readable record)
- `contexts/<qid>.txt`
- `outputs/<qid>.md`
- `retrieved/<qid>.json`

4) Generate a labeling sheet (precision@k proxy):

```powershell
python -m quokka_eval.make_labels --run .\runs\<run_id>\run.jsonl
```

5) After you label the CSV, compute scores:

```powershell
python -m quokka_eval.score --run .\runs\<run_id>\run.jsonl --labels .\runs\<run_id>\labels.csv
```

## Notes
- This harness **never uses web data**.
- The default `mock` pipeline is deterministic and **extractive-only** (no hallucination).
- PDF extraction is best-effort (optional dependency). For early testing, `.txt`/`.md`/`.docx` are easiest.
