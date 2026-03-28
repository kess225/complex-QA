"""
table_extractor.py
------------------
Extract tables from the NASA handbook PDF as structured markdown.
Tables are critical for NASA handbook (entry criteria, checklists, reviews).

Usage:
    from backend.ingestion.table_extractor import TableExtractor
    te = TableExtractor('data/raw/nasa_handbook.pdf')
    tables = te.extract_all_tables()
    te.save_json('backend/extracted/table_metadata.json')
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pdfplumber


class TableExtractor:
    """
    Extracts all tables from a PDF and converts them to markdown.
    Provides lookup by page number and keyword search.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = str(pdf_path)
        self.tables: List[Dict] = []
        self.table_metadata: List[Dict] = []
        self._extracted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all_tables(self) -> List[Dict]:
        """
        Extract all tables from the PDF with metadata.

        Returns:
            List of dicts: {page, table_idx, markdown, raw, caption}
        """
        self.tables = []
        self.table_metadata = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Get table objects for bounding-box access
                table_objects = page.find_tables()
                raw_tables = page.extract_tables()

                if not raw_tables:
                    continue

                for table_idx, (table_obj, raw_table) in enumerate(
                    zip(table_objects, raw_tables)
                ):
                    if not raw_table:
                        continue

                    # Sanitise: replace None with empty string
                    clean_raw = [
                        [cell if cell is not None else "" for cell in row]
                        for row in raw_table
                    ]

                    md = self._to_markdown(clean_raw)
                    caption = self._extract_caption(page, page_num, table_idx, table_obj)

                    record = {
                        "page": page_num,
                        "table_idx": table_idx,
                        "markdown": md,
                        "raw": clean_raw,
                        "caption": caption,
                    }
                    self.tables.append(record)
                    self.table_metadata.append(
                        {
                            "page": page_num,
                            "table_idx": table_idx,
                            "rows": len(clean_raw),
                            "cols": len(clean_raw[0]) if clean_raw else 0,
                            "caption": caption,
                        }
                    )

        self._extracted = True
        return self.tables

    def save_json(self, output_path: str) -> None:
        """Persist table metadata (without raw data) to JSON for later retrieval."""
        output = {
            "tables": self.table_metadata,
            "total": len(self.table_metadata),
            "file": self.pdf_path,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def get_table_by_page(self, page_num: int) -> List[Dict]:
        """Return all tables found on a given page (1-indexed)."""
        return [t for t in self.tables if t["page"] == page_num]

    def find_tables_with_keyword(self, keyword: str) -> List[Dict]:
        """Return tables whose caption or raw content contains *keyword*."""
        kw = keyword.lower()
        return [
            t
            for t in self.tables
            if kw in t["caption"].lower() or kw in str(t["raw"]).lower()
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_markdown(self, table: List[List[str]]) -> str:
        """Convert a list-of-lists table to a GitHub-flavoured markdown table."""
        if not table or not table[0]:
            return ""
        try:
            df = pd.DataFrame(table[1:], columns=table[0])
            return df.to_markdown(index=False)
        except Exception:
            # Fallback: plain pipe-separated text
            rows = []
            for i, row in enumerate(table):
                rows.append("| " + " | ".join(str(c) for c in row) + " |")
                if i == 0:
                    rows.append(
                        "| " + " | ".join("---" for _ in row) + " |"
                    )
            return "\n".join(rows)

    def _extract_caption(
        self, page, page_num: int, table_idx: int, table_obj
    ) -> str:
        """
        Heuristic: look for 'Table X' text near the table bounding box.
        Falls back to 'Table on page N (index I)'.
        """
        fallback = f"Table on page {page_num} (index {table_idx})"
        try:
            bbox = table_obj.bbox  # (x0, top, x1, bottom)
            # Search for text just above the table (within 40 pts)
            above_bbox = (0, max(0, bbox[1] - 40), page.width, bbox[1])
            above_words = page.crop(above_bbox).extract_text() or ""
            # Search for text just below the table (within 30 pts)
            below_bbox = (0, bbox[3], page.width, min(page.height, bbox[3] + 30))
            below_words = page.crop(below_bbox).extract_text() or ""

            for text in (above_words, below_words):
                match = re.search(
                    r"Table\s+[A-Z0-9][A-Z0-9\-\.]*", text, re.IGNORECASE
                )
                if match:
                    return match.group(0).strip()
        except Exception:
            pass
        return fallback


# ------------------------------------------------------------------
# CLI helper
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    BACKEND_ROOT = Path(__file__).resolve().parents[1]
    DEFAULT_PDF = BACKEND_ROOT / "data" / "raw" / "nasa_handbook.pdf"
    DEFAULT_OUT = BACKEND_ROOT / "extracted" / "table_metadata.json"

    parser = argparse.ArgumentParser(description="Extract tables from NASA handbook PDF")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="Path to PDF")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSON path")
    args = parser.parse_args()

    te = TableExtractor(args.pdf)
    tables = te.extract_all_tables()
    print(f"Found {len(tables)} tables")
    te.save_json(args.out)
    print(f"Saved metadata → {args.out}")
