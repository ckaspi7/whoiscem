"""Section-aware chunking for Markdown.

The semantic chunker split the whole resume into three chunks, one of them 4,100
characters, so top-3 reranking returned the entire document on every query: RRF
had nothing to fuse and the cross-encoder nothing to discriminate between.

Markdown already carries the structure a resume has — sections, roles, bullets —
so this uses the headings instead of guessing at embedding distance. Each chunk
keeps its heading path, which gives the embedding something to anchor on and
tells the model which role a bullet belongs to.
"""

from __future__ import annotations

import re

from retrieval.types import TextChunk

# Large enough to hold a role's bullets together, small enough that top-3 is a
# selection rather than the whole document.
MAX_CHARS = 700
MIN_CHARS = 120

_H2 = re.compile(r"^## +(.+?)\s*$", re.M)
_H3 = re.compile(r"^### +(.+?)\s*$", re.M)


def chunk_markdown(text: str, max_chars: int = MAX_CHARS) -> list[TextChunk]:
    """Split Markdown into chunks that respect its heading structure."""
    chunks: list[TextChunk] = []
    for section, body in _split_sections(text):
        for block_title, block in _split_subsections(body):
            heading = f"{section} — {block_title}" if block_title else section
            for piece in _pack(block, max_chars):
                chunks.append(TextChunk(text=f"{heading}\n{piece}", section=section))
    return _merge_runts(chunks, max_chars)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split on H2, keeping any preamble under the document title."""
    matches = list(_H2.finditer(text))
    if not matches:
        return [("", text.strip())]

    sections = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        title = preamble.splitlines()[0].lstrip("# ").strip()
        sections.append((title or "Overview", preamble))

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((match.group(1), text[match.end() : end].strip()))
    return sections


def _split_subsections(body: str) -> list[tuple[str, str]]:
    """Split a section on H3 — one per role, in the experience section."""
    matches = list(_H3.finditer(body))
    if not matches:
        return [("", body)]

    blocks = []
    lead = body[: matches[0].start()].strip()
    if lead:
        blocks.append(("", lead))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        blocks.append((match.group(1), body[match.end() : end].strip()))
    return blocks


def _pack(block: str, max_chars: int) -> list[str]:
    """Greedily pack paragraphs and bullets up to max_chars.

    Splits only at blank lines or between top-level bullets, so a bullet is
    never cut in half — half a claim retrieves as well as a whole one and reads
    far worse in an answer.
    """
    block = block.strip()
    if not block:
        return []
    if len(block) <= max_chars:
        return [block]

    pieces: list[str] = []
    current: list[str] = []
    length = 0

    for unit in _units(block):
        unit_len = len(unit) + 1
        if current and length + unit_len > max_chars:
            pieces.append("\n".join(current).strip())
            current, length = [], 0
        current.append(unit)
        length += unit_len

    if current:
        pieces.append("\n".join(current).strip())
    return [p for p in pieces if p]


def _units(block: str) -> list[str]:
    """Break a block into bullets and paragraphs, keeping continuations attached."""
    units: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        starts_unit = stripped.startswith(("- ", "* ", "| ", "**")) or not units
        if starts_unit or not stripped:
            units.append(line)
        else:
            # An indented continuation of the previous bullet.
            units[-1] = f"{units[-1]}\n{line}"
    return [u for u in units if u.strip()]


def _heading_of(chunk: TextChunk) -> str:
    return chunk.text.split("\n", 1)[0]


def _merge_runts(chunks: list[TextChunk], max_chars: int) -> list[TextChunk]:
    """Fold tiny chunks into the previous one under the same heading.

    A heading with one short line under it carries little signal alone and
    dilutes the ranking. Merging is keyed on the full heading path, not the
    section: a short role folded into the role above it would attribute one
    employer's bullets to another, which is the worst answer a resume bot can
    give — confidently wrong about who did what.
    """
    merged: list[TextChunk] = []
    for chunk in chunks:
        if (
            merged
            and len(chunk.text) < MIN_CHARS
            and _heading_of(merged[-1]) == _heading_of(chunk)
            and len(merged[-1].text) + len(chunk.text) <= max_chars
        ):
            body = chunk.text.split("\n", 1)[1] if "\n" in chunk.text else ""
            merged[-1] = TextChunk(text=f"{merged[-1].text}\n{body}".rstrip(), section=chunk.section)
        else:
            merged.append(chunk)
    return merged
