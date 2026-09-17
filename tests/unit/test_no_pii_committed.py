"""Fails if personal data is tracked by git.

The previous repository published an email address, a phone number and family
details for eighteen months because the protection was a .gitignore entry and a
sentence in a README. This test is the protection now: it reads what git
actually tracks, not what the ignore file intends.

Lives under tests/unit/ rather than tests/ so that CI's `pytest tests/unit/`
runs it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths that must never be tracked, whatever .gitignore currently says.
FORBIDDEN_PATHS = (
    re.compile(r"\.db$"),
    re.compile(r"(^|/)seed_data\.json$"),
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)secrets\.toml$"),
)

TEXT_SUFFIXES = {
    ".py", ".md", ".json", ".toml", ".yml", ".yaml", ".txt", ".cfg", ".ini",
    ".example", ".sh", ".ps1", ".html", ".csv",
}

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"(?:\+\d{1,2}[\s.-]?)?\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")

# Documentation and fixtures legitimately contain address-shaped text.
PLACEHOLDER_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "email.com")
PLACEHOLDER_EMAIL_LOCALS = ("you", "your", "user", "noreply", "test")

# This file quotes realistic contact details on purpose, to prove the scanner
# fires. Scanning it would flag its own fixtures.
SELF = "tests/unit/test_no_pii_committed.py"


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git unavailable: {exc}")
    if out.returncode != 0:
        pytest.skip("not a git repository")
    return [line for line in out.stdout.splitlines() if line.strip()]


def _is_placeholder_email(match: str) -> bool:
    local, _, domain = match.partition("@")
    return domain.lower().endswith(PLACEHOLDER_EMAIL_DOMAINS) or local.lower() in PLACEHOLDER_EMAIL_LOCALS


def _is_placeholder_phone(match: str) -> bool:
    """Only two things count as obviously fake.

    Substring matching on "555" or "0000" was the first attempt and it is wrong:
    a real number can contain either, so the guard would wave it through.
    """
    digits = re.sub(r"\D", "", match)
    local = digits[-10:]  # drop any country code before judging
    if len(set(local)) == 1:  # 000-000-0000 and friends
        return True
    # 555-0100 through 555-0199: the range reserved for fiction.
    return bool(re.search(r"555\s?01\d{2}$", digits))


def _pdf_text(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError:  # pragma: no cover
        pytest.skip("pdfplumber not installed")
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _scan(text: str) -> list[str]:
    hits = [m for m in EMAIL.findall(text) if not _is_placeholder_email(m)]
    hits += [m for m in PHONE.findall(text) if not _is_placeholder_phone(m)]
    return hits


def test_no_personal_data_files_are_tracked():
    offenders = [
        f for f in _tracked_files() if any(pattern.search(f) for pattern in FORBIDDEN_PATHS)
    ]
    assert not offenders, f"personal data files are tracked by git: {offenders}"


def test_no_contact_details_in_tracked_text():
    offenders: dict[str, list[str]] = {}
    for rel in _tracked_files():
        path = REPO_ROOT / rel
        if rel == SELF or path.suffix.lower() not in TEXT_SUFFIXES or not path.exists():
            continue
        hits = _scan(path.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            offenders[rel] = sorted(set(hits))

    assert not offenders, f"contact details found in tracked files: {offenders}"


def test_no_contact_details_in_tracked_pdfs():
    """Guards the redacted resume: a PDF hides its text from every other check."""
    pdfs = [REPO_ROOT / f for f in _tracked_files() if f.lower().endswith(".pdf")]
    if not pdfs:
        pytest.skip("no PDFs tracked yet")

    offenders = {p.name: sorted(set(_scan(_pdf_text(p)))) for p in pdfs}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"contact details found in tracked PDFs: {offenders}"


# A guard nobody has seen fail is a guard nobody should trust.
@pytest.mark.parametrize("sample", [
    "reach me at firstname.lastname@gmail.com",
    "CEM KASPI (778) 839-0000 Vancouver, BC",
    "call 604-555-0000 x2",
    "direct line 236.555.9876",
])
def test_scanner_catches_realistic_contact_details(sample):
    assert _scan(sample), f"scanner missed {sample!r}"


@pytest.mark.parametrize("sample", [
    "OPENAI_API_KEY=sk-...",
    "you@example.com",
    "+1 (000) 000-0000",
    "+1 236 555 0100",  # the range reserved for fiction
    "see docs at https://arize.com/docs/phoenix",
])
def test_scanner_ignores_placeholders_and_docs(sample):
    assert not _scan(sample), f"scanner false-positived on {sample!r}"
