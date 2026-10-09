"""retrieval/fallback.py: the resume served from disk with no network."""

from __future__ import annotations

import pytest

from retrieval.fallback import DiskResumeStore

_RESUME_MD = """# Someone

## Education

BASc, Electrical Engineering, University of British Columbia, 2015 - 2021.
Minor in Entrepreneurship. Dean's List 2019-2021.

## Experience

### TELUS Communications - AI/ML Engineer

Built a production RAG agent for annual-report questions and its evaluation suite.
"""


def _store(tmp_path) -> DiskResumeStore:
    resume_file = tmp_path / "resume.md"
    resume_file.write_text(_RESUME_MD, encoding="utf-8")
    return DiskResumeStore(str(resume_file))


def test_it_chunks_the_resume_with_the_same_structural_chunker_the_index_uses(tmp_path):
    store = _store(tmp_path)

    chunks = store.get_all_chunks()

    assert len(chunks) >= 2
    assert {chunk.section for chunk in chunks} >= {"Education", "Experience"}
    assert store.corpus_chars == sum(len(chunk.text) for chunk in chunks)


def test_dense_search_degrades_to_keyword_ranking(tmp_path):
    store = _store(tmp_path)

    top = store.dense_search("electrical engineering degree", top_k=1)

    assert top and "Electrical Engineering" in top[0].text


def test_a_pdf_resume_is_rejected_rather_than_chunked_badly(tmp_path):
    pdf = tmp_path / "resume.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with pytest.raises(ValueError, match="Markdown"):
        DiskResumeStore(str(pdf))
