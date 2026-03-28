"""
acronym_extractor.py
--------------------
One-time ingestion script that parses Appendix A (PDF pages 215–219) of the
NASA Systems Engineering Handbook and writes a flat acronym→expansion dictionary
to backend/extracted/acronyms.json.

Run:
        python backend/ingestion/acronym_extractor.py
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pdfplumber

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BACKEND_ROOT / "extracted" / "acronyms.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_ACRONYM_RE = re.compile(r"^[A-Z0-9/\-]+$")


def _clean_text(value: str) -> str:
    """Normalize internal whitespace in extracted strings."""
    return re.sub(r"\s+", " ", value.strip())


def _valid_acronym(key: str, value: str) -> bool:
    """Return True when (key, value) looks like a real acronym entry.

    Args:
        key:   Candidate acronym string (already stripped).
        value: Candidate expansion string (already stripped).

    Returns:
        True if both fields pass all validation rules, False otherwise.
    """
    if not key or not value:
        return False
    if not (1 <= len(key) <= 12):
        return False
    if not _ACRONYM_RE.match(key.upper()):
        return False
    if len(value) <= len(key):
        return False
    if key[0].isdigit() or value[0].isdigit():
        return False
    return True


def _parse_line(line: str) -> tuple[str, str] | None:
    """Split a plain-text line into (acronym, expansion) if it matches.

    Splits on the first occurrence of two or more consecutive spaces or a tab,
    treating the left token as the acronym and the remainder as the expansion.

    Args:
        line: A single line of text from pdfplumber's extract_text().

    Returns:
        A (acronym_upper, expansion) tuple, or None if the line does not match.
    """
    raw = line.strip()
    if not raw:
        return None

    # Primary split: tabs or 2+ spaces, as defined by the ingestion contract.
    parts = re.split(r"\t|  +", raw, maxsplit=1)
    if len(parts) >= 2:
        key = _clean_text(parts[0])
        value = _clean_text(parts[1])
        if _valid_acronym(key.upper(), value):
            return key.upper(), value

    # Real NASA appendix lines are often "ACRONYM Expansion" with one space.
    # Use first token as acronym and keep the rest as expansion.
    m = re.match(r"^([A-Z0-9/\-]+)\s+(.+)$", raw)
    if not m:
        return None
    key = _clean_text(m.group(1))
    value = _clean_text(m.group(2))
    if not _valid_acronym(key.upper(), value):
        return None
    return key.upper(), value


def _extract_row_pairs(row: list[str | None]) -> list[tuple[str, str]]:
    """Return valid (acronym, expansion) pairs from adjacent table columns.

    Args:
        row: Raw table row emitted by pdfplumber.

    Returns:
        A list of valid (acronym_upper, expansion) pairs found in the row.
    """
    values = [_clean_text(cell or "") for cell in row]
    pairs: list[tuple[str, str]] = []
    for i in range(len(values) - 1):
        key = values[i]
        value = values[i + 1]
        if _valid_acronym(key.upper(), value):
            pairs.append((key.upper(), value))
    return pairs


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_acronyms(pdf_path: Path) -> dict[str, str]:
    """Parse Appendix A (pages 215–219) and return a flat acronym dictionary.

    Args:
        pdf_path: Absolute or relative path to the NASA handbook PDF.

    Returns:
        A dict mapping each uppercase acronym string to its full expansion.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        Exception: Re-raised after logging if pdfplumber fails to open/read the PDF.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"NASA handbook PDF not found at: {pdf_path}. "
            "Download it and place it at data/raw/nasa_handbook.pdf before running."
        )

    acronyms: dict[str, str] = {}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[214:219]:  # 0-indexed → PDF pages 215-219 inclusive
                page_added = 0

                # --- Strategy A: structured table rows ---
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row and len(row) >= 2:
                            for key, value in _extract_row_pairs(row):
                                acronyms[key.upper()] = value
                                page_added += 1

                # --- Strategy B: plain text lines fallback if table parsing found none ---
                if page_added == 0:
                    text = page.extract_text() or ""
                    for line in text.splitlines():
                        parsed = _parse_line(line)
                        if parsed:
                            acronyms[parsed[0]] = parsed[1]
    except Exception:
        logger.exception("pdfplumber failed while reading %s", pdf_path)
        raise

    logger.info("Extracted %d acronyms from %s", len(acronyms), pdf_path)
    return acronyms


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save(acronyms: dict[str, str], path: Path = OUTPUT_PATH) -> None:
    """Write the acronym dictionary to *path* as UTF-8 JSON.

    Overwrites any existing file (idempotent).

    Args:
        acronyms: Flat dict mapping uppercase acronym strings to expansions.
        path:     Destination path (default: data/extracted/acronyms.json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(acronyms, f, indent=2, ensure_ascii=False, sort_keys=True)
    logger.info("Wrote %d acronyms → %s", len(acronyms), path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract NASA Appendix A acronyms.")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=BACKEND_ROOT / "data" / "raw" / "nasa_handbook.pdf",
        help="Path to nasa_handbook.pdf",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    PDF_PATH = args.pdf
    acronyms = extract_acronyms(PDF_PATH)
    save(acronyms)
