"""Chunking and the identifier join between the dense and sparse indexes."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrieval.bm25 import BM25Index
from retrieval.chunking import MAX_CHARS, chunk_markdown
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.types import ScoredChunk

RESUME = Path(__file__).resolve().parents[2] / "data" / "resume.md"

SAMPLE = """# Jane Doe

City, Country

## Summary

An engineer.

## Experience

### ACME — Staff Engineer
**2020 – Present**

- Built the thing that does the stuff, and then built a second thing which was
  considerably larger than the first thing and needed more explanation.
- Ran the other thing.

### Globex — Engineer
**2018 – 2020**

- Did a job.

## Education

**BSc, Computer Science** — Some University
"""


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_chunks_follow_the_heading_structure():
    chunks = chunk_markdown(SAMPLE)
    sections = {c.section for c in chunks}
    assert {"Summary", "Experience", "Education"} <= sections


def test_every_chunk_records_its_section():
    """section was plumbed through four files and never populated."""
    for chunk in chunk_markdown(SAMPLE):
        assert chunk.section, f"chunk without a section: {chunk.text[:40]!r}"


def test_chunks_carry_their_heading_path():
    chunks = chunk_markdown(SAMPLE)
    experience = [c for c in chunks if c.section == "Experience"]
    assert any("ACME" in c.text for c in experience)
    assert any("Globex" in c.text for c in experience)


def test_roles_do_not_bleed_into_each_other():
    """A bullet attributed to the wrong employer is worse than no answer."""
    for chunk in chunk_markdown(SAMPLE):
        assert not ("ACME" in chunk.text and "Globex" in chunk.text)


def test_chunks_respect_the_size_ceiling():
    for chunk in chunk_markdown(SAMPLE):
        assert len(chunk.text) <= MAX_CHARS + 200, f"{len(chunk.text)} chars"


def test_bullets_are_never_split_mid_sentence():
    for chunk in chunk_markdown(SAMPLE, max_chars=120):
        for line in chunk.text.splitlines():
            if line.strip().startswith("- "):
                assert not line.rstrip().endswith(("and", "the", "a", "of", "which"))


def test_empty_and_headingless_input_are_handled():
    assert chunk_markdown("") == []
    plain = chunk_markdown("Just a sentence with no headings at all.")
    assert len(plain) == 1


@pytest.mark.skipif(not RESUME.exists(), reason="resume.md not present")
def test_the_real_resume_produces_a_usable_corpus():
    """Three chunks made top-3 reranking a no-op: it returned everything."""
    chunks = chunk_markdown(RESUME.read_text(encoding="utf-8"))
    assert len(chunks) >= 10, f"only {len(chunks)} chunks — reranking cannot discriminate"
    assert max(len(c.text) for c in chunks) <= MAX_CHARS + 200


# ---------------------------------------------------------------------------
# The identifier join
# ---------------------------------------------------------------------------


def test_bm25_returns_the_identifiers_it_was_given():
    """Not list positions.

    BM25 keyed on enumeration order while the vector store keyed on Qdrant
    point ids, and fusion joined them on string equality. That worked only
    while ids happened to be assigned in list order.
    """
    chunks = [
        ScoredChunk(chunk_id="900", text="Cem works at TELUS in Vancouver.", score=0.0),
        ScoredChunk(chunk_id="17", text="Cem co-founded NeoWise, a wearable startup.", score=0.0),
        ScoredChunk(chunk_id="404", text="Cem studied electrical engineering at UBC.", score=0.0),
    ]
    index = BM25Index()
    index.build(chunks)

    results = index.search("NeoWise wearable startup", top_k=3)
    assert results
    assert results[0].chunk_id == "17", f"got {results[0].chunk_id}"
    assert {r.chunk_id for r in results} <= {"900", "17", "404"}


def test_fusion_joins_dense_and_sparse_on_non_sequential_ids():
    """The end-to-end shape of the bug: same chunk, both lists, one entry."""
    chunks = [
        ScoredChunk(chunk_id="900", text="Cem works at TELUS in Vancouver.", score=0.0),
        ScoredChunk(chunk_id="17", text="Cem co-founded NeoWise, a wearable startup.", score=0.0),
    ]
    index = BM25Index()
    index.build(chunks)

    sparse = index.search("NeoWise", top_k=5)
    dense = [ScoredChunk(chunk_id="17", text=chunks[1].text, score=0.9)]

    fused = reciprocal_rank_fusion(dense, sparse)
    assert len(fused) == 1, f"the same chunk fused as {len(fused)} entries: {[c.chunk_id for c in fused]}"
    assert fused[0].chunk_id == "17"


def test_bm25_preserves_section_metadata():
    # More than one document on purpose: Okapi gives a term present in every
    # document a non-positive IDF, and the index drops non-positive scores.
    chunks = [
        ScoredChunk(
            chunk_id="3", text="Studied electrical engineering at UBC.", score=0.0, section="Education"
        ),
        ScoredChunk(chunk_id="4", text="Worked at TELUS in Vancouver.", score=0.0, section="Experience"),
        ScoredChunk(chunk_id="5", text="Listens to hip hop.", score=0.0, section="Music"),
    ]
    index = BM25Index()
    index.build(chunks)

    results = index.search("UBC")
    assert results, "a term unique to one chunk should be findable"
    assert results[0].section == "Education"


def test_bm25_on_an_empty_corpus_returns_nothing():
    index = BM25Index()
    index.build([])
    assert index.search("anything") == []
