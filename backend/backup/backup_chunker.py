import fitz
import pdfplumber
import os
import json
from pathlib import Path
import pandas as pd
import re
from langchain_core.documents import Document

# ---------------- CONFIG ----------------
CHUNK_SIZE = 800

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = str(_BACKEND_ROOT / "data" / "raw" / "rag_structure_csv.csv")
OUTPUT_DIR = str(_BACKEND_ROOT / "extracted")

noise = ["NASA Systems Engineering Handbook "]

HEADER_RATIO = 0.1
FOOTER_RATIO = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)


# ---------------- BOOLEAN ----------------
def to_bool(val):
    return str(val).strip().lower() in ["true", "1", "yes"]


# ---------------- LOAD CSV ----------------
def load_sections(csv_path, total_pages):
    df = pd.read_csv(csv_path)
    sections = []

    for _, row in df.iterrows():
        if not to_bool(row["include_in_rag"]):
            continue

        start = int(row["from_page"]) - 1
        end = int(row["to_page"]) - 1

        if start >= total_pages:
            continue

        sections.append({
            "title": row["Title"],
            "from_page": max(0, start),
            "to_page": min(end, total_pages - 1)
        })

    print(f"✅ Sections loaded: {len(sections)}")
    return sections


# ---------------- CLEANING ----------------
def normalize_pdf_text(text):
    replacements = {
        "ﬂ": "fl", "ﬁ": "fi", "ﬀ": "ff",
        "ﬃ": "ffi", "ﬄ": "ffl",
        "–": "-", "—": "-",
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "\xa0": " "
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def clean_text(text):
    text = re.sub(r"\.{3,}\s*\d+", "", text)
    return " ".join(text.split())


def is_header_footer(bbox, h):
    return bbox[1] < h * HEADER_RATIO or bbox[3] > h * (1 - FOOTER_RATIO)


def remove_noise(text, bbox, h):
    if is_header_footer(bbox, h):
        for n in noise:
            text = text.replace(n, "")
    return text


# ---------------- EXTRACTION ----------------
def extract_text_blocks(doc, pages):
    elements = []

    for p in pages:
        page = doc[p]
        h = page.rect.height

        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue

            text = " ".join(
                span["text"]
                for line in block["lines"]
                for span in line["spans"]
            )

            text = normalize_pdf_text(text)
            text = remove_noise(text, block["bbox"], h)
            text = clean_text(text)

            if text:
                elements.append({
                    "type": "text",
                    "page": p,
                    "bbox": block["bbox"],
                    "content": text
                })

    return elements


def extract_tables(pdf_path, pages):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pages:
            for t in pdf.pages[p].extract_tables():
                tables.append({
                    "type": "table",
                    "page": p,
                    "bbox": pdf.pages[p].bbox,
                    "content": t
                })
    return tables


# ---------------- IMAGE EXTRACTION (FIXED) ----------------
def extract_images(doc, pages):
    images = []

    for p in pages:
        page = doc[p]

        # -------- RAW IMAGES --------
        for i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)

            path = os.path.join(
                IMAGE_DIR, f"page{p}_img{i}.{base['ext']}"
            )

            with open(path, "wb") as f:
                f.write(base["image"])

            images.append({
                "type": "image",
                "page": p,
                "bbox": None,
                "content": path,
                "source": "raw"
            })

        # -------- HIGH-RES PAGE RENDER --------
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        page_img_path = os.path.join(IMAGE_DIR, f"page{p}_full.png")
        pix.save(page_img_path)

        images.append({
            "type": "image",
            "page": p,
            "bbox": None,
            "content": page_img_path,
            "source": "rendered"
        })

    return images


# ---------------- MERGE ----------------
def merge_elements(texts, tables, images):
    elements = texts + tables + images
    elements.sort(key=lambda x: (x["page"], x["bbox"][1] if x["bbox"] else 0))
    return elements


# ---------------- RELATIONSHIP ----------------
def is_caption(text):
    return text.lower().startswith(("figure", "table"))


def find_context(elements, idx, window=5):
    ctx = []
    for i in range(max(0, idx - window), min(len(elements), idx + window)):
        if elements[i]["type"] == "text":
            ctx.append(elements[i]["content"])
    return " ".join(ctx)


def attach_relationships(elements):
    for i, e in enumerate(elements):

        if e["type"] in ["table", "image"]:
            e["context"] = find_context(elements, i)

        if e["type"] == "text" and is_caption(e["content"]):
            if i + 1 < len(elements) and elements[i+1]["type"] in ["image", "table"]:
                elements[i+1]["caption"] = e["content"]

    return elements


# ---------------- TABLE ----------------
def table_to_text(table):
    return "\n".join([" | ".join([str(c) for c in row]) for row in table])


# ---------------- CHUNKING ----------------
def create_chunks(elements, section):
    chunks = []

    text, tables, images = "", [], []
    start_page = None

    for e in elements:

        if e["type"] == "text":
            if not text:
                start_page = e["page"]

            if len(text) + len(e["content"]) < CHUNK_SIZE:
                text += " " + e["content"]
            else:
                chunks.append({
                    "text": text.strip(),
                    "tables": tables,
                    "images": images,
                    "page_start": start_page,
                    "page_end": e["page"]
                })
                text, tables, images = e["content"], [], []
                start_page = e["page"]

        elif e["type"] == "table":
            t = table_to_text(e["content"])

            if "caption" in e:
                t = e["caption"] + "\n" + t

            if "context" in e:
                t = e["context"] + "\n\n" + t

            tables.append(t)

        elif e["type"] == "image":
            img_text = f"Image: {e['content']}"

            if "caption" in e:
                img_text = e["caption"] + "\n" + img_text

            if "context" in e:
                img_text = e["context"] + "\n\n" + img_text

            images.append(img_text)

    if text:
        chunks.append({
            "text": text.strip(),
            "tables": tables,
            "images": images,
            "page_start": start_page,
            "page_end": e["page"]
        })

    return chunks


# ---------------- DOCUMENT ----------------
def create_documents(chunks, section):
    docs = []

    for i, c in enumerate(chunks):
        content = f"{section['title']}\n\n{c['text']}"

        if c["tables"]:
            content += "\n\nTables:\n" + "\n\n".join(c["tables"])

        if c["images"]:
            content += "\n\nImages:\n" + "\n\n".join(c["images"])

        metadata = {
            "chunk_id": i,
            "title": section["title"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "has_table": bool(c["tables"]),
            "has_image": bool(c["images"]),
            "num_tables": len(c["tables"]),
            "num_images": len(c["images"])
        }

        docs.append(Document(page_content=content, metadata=metadata))

    return docs


# ---------------- MAIN ----------------
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    print(f"📄 Total Pages: {total_pages}")

    sections = load_sections(CSV_PATH, total_pages)
    all_docs = []

    for section in sections:
        print(f"Processing: {section['title']}")

        pages = list(range(section["from_page"], section["to_page"] + 1))

        texts = extract_text_blocks(doc, pages)
        tables = extract_tables(pdf_path, pages)
        images = extract_images(doc, pages)

        elements = merge_elements(texts, tables, images)
        elements = attach_relationships(elements)

        chunks = create_chunks(elements, section)
        docs = create_documents(chunks, section)

        all_docs.extend(docs)

    output_file = os.path.join(OUTPUT_DIR, "documents.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            [{"content": d.page_content, "metadata": d.metadata} for d in all_docs],
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\n✅ Saved {len(all_docs)} docs → {output_file}")
    return all_docs


# ---------------- RUN ----------------
pdf_path = str(_BACKEND_ROOT / "data" / "raw" / "nasa_handbook.pdf")

docs = process_pdf(pdf_path)

print(docs[0])
print(f"Total Documents: {len(docs)}")