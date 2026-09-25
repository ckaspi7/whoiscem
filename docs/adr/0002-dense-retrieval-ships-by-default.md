# 0002 — Dense retrieval ships by default; the hybrid pipeline is measured, not removed

**Status:** Accepted

## Context

The project's centerpiece was a hybrid retrieval pipeline: dense search
(Qdrant + OpenAI embeddings), sparse search (BM25), Reciprocal Rank Fusion of
the two, then a cross-encoder rerank to the final top-N. It was built and
shipped without ever being measured against the alternative of just using
one of its parts. `RETRIEVAL_STRATEGY` did not exist as a setting; the
pipeline always ran in full.

## Decision

Build `eval/ablate_retrieval.py`, measure every strategy against the golden
references with routing excluded, and default to whichever wins — not to the
one the architecture diagram implies is more sophisticated.

| variant | recall@k | MRR | ms/query |
|---|---|---|---|
| **dense only, k=5** | **100.0%** | **0.619** | 215 |
| RRF fusion, k=5 | 83.3% | 0.484 | 237 |
| RRF + cross-encoder, k=5 | 91.7% | 0.540 | 940 |
| whole document | 100.0% | 0.550 | 1,216 |

Plain dense search beats the full hybrid pipeline outright — better recall,
better MRR, 4.4× lower latency than the reranked variant. BM25 contributes
almost nothing because a one-page resume has too few documents for term
frequency to separate anything, and the cross-encoder spends ~700ms/query
demoting chunks dense search had already ranked first. Separately, retrieval
selection *of any kind* is a net loss on this corpus: grounding improves
monotonically with the amount of context returned, because the model
benefits from more than the one snippet a question references.

`RETRIEVAL_STRATEGY` defaults to `auto`: pass the whole corpus while it fits
in a prompt (`CORPUS_FITS_CONTEXT_CHARS`), fall back to dense search once it
does not.

## Consequences

- The hybrid pipeline (BM25, RRF, the cross-encoder) is not deleted. It is
  kept, fully wired, and selectable via `RETRIEVAL_STRATEGY` — this ranking
  is a property of a 7KB corpus, not a law, and is expected to change once
  the corpus grows past the point where "the whole document" stops fitting.
  Removing it would mean rebuilding it later instead of just flipping a
  setting.
- `LIMITATIONS.md` states plainly that the project's own flagship feature
  loses to the simpler alternative on the corpus it currently has, rather
  than only reporting the numbers that flatter the more complex design.
- MRR and context precision are rank-sensitive metrics that depend on chunk
  granularity; `eval/compare.py` already treats a rechunk (and, separately, an
  `agent_mode` change) as changing what a "chunk" is and does not gate those
  metrics across either — the same discipline this ablation's own numbers
  are read under.
- Growing the corpus (project dossiers, an engineering notes log) is the
  actual next step for this pipeline to demonstrate anything, not further
  retrieval tuning on a document this small.
