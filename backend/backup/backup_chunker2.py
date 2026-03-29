import fitz
import pdfplumber
import os
import json
import pandas as pd
import re
from langchain_core.documents import Document
import tiktoken  # For token counting

# =============== CONFIG ===============
# Token-based chunking (not character-based)
MAX_TOKENS = 1000
OVERLAP_PCT = 0.15
OVERLAP_TOKENS = int(MAX_TOKENS * OVERLAP_PCT)  # 150 tokens

# Token encoding
ENCODING = "cl100k_base"  # GPT/Claude standard encoding

CSV_PATH = "C:\\rag\\data\\raw\\rag_structure_csv.csv"
OUTPUT_DIR = "C:\\rag\\backend\\extracted"

os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# NOISE REMOVAL CONFIG
NOISE_VARIABLES = [
    "NASA Systems Engineering Handbook Rev 2",
    "NASA Systems Engineering Handbook",
    "Rev 2",
    "Page [0-9]+",
    "^\\s*$",  # Empty lines
]


# =============== TOKEN COUNTER ===============
class TokenCounter:
    """
    Efficient token counter using tiktoken.
    Caches the encoding to avoid reloading.
    """
    _encoder = None
    
    @classmethod
    def get_encoder(cls):
        if cls._encoder is None:
            cls._encoder = tiktoken.get_encoding(ENCODING)
        return cls._encoder
    
    @classmethod
    def count_tokens(cls, text):
        """Count tokens in text."""
        if not text or not isinstance(text, str):
            return 0
        encoder = cls.get_encoder()
        return len(encoder.encode(text))
    
    @classmethod
    def encode(cls, text):
        """Get token IDs for text."""
        if not text or not isinstance(text, str):
            return []
        encoder = cls.get_encoder()
        return encoder.encode(text)


# =============== UTILS ===============
def to_bool(val):
    return str(val).strip().lower() in ["true", "1", "yes"]


def clean_text(text):
    return " ".join(text.split())


def remove_noise(text, noise_vars=None):
    """
    Remove noise patterns from text using a list of noise variables.
    
    Args:
        text (str): The text to clean
        noise_vars (list): List of noise patterns (exact matches or regex patterns)
    
    Returns:
        str: Cleaned text with noise removed
    """
    if noise_vars is None:
        noise_vars = NOISE_VARIABLES
    
    cleaned = text
    
    for noise_pattern in noise_vars:
        try:
            # Try as regex pattern first
            cleaned = re.sub(noise_pattern, "", cleaned, flags=re.IGNORECASE)
        except re.error:
            # If regex fails, treat as exact string match
            cleaned = cleaned.replace(noise_pattern, "")
    
    # Final cleanup: remove extra whitespace
    cleaned = " ".join(cleaned.split())
    
    return cleaned


def is_noise(text, noise_vars=None):
    """
    Check if text is entirely noise.
    
    Args:
        text (str): The text to check
        noise_vars (list): List of noise patterns
    
    Returns:
        bool: True if text is noise, False otherwise
    """
    if noise_vars is None:
        noise_vars = NOISE_VARIABLES
    
    cleaned = remove_noise(text, noise_vars)
    return len(cleaned.strip()) == 0


# =============== COLOR ===============
def is_blue(color_int):
    if color_int is None:
        return False

    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255

    return (
        b > 120 and
        b > r * 1.3 and
        b > g * 1.3 and
        g < 140
    )


# =============== CAPTION (STRICT) ===============
def get_caption_type(text, is_blue_text):
    text = text.lower().strip()

    if not is_blue_text:
        return None

    if re.match(r"^figure\s+\d+", text):
        return "image"

    if re.match(r"^table\s+\d+", text):
        return "table"

    return None


# =============== LOAD CSV ===============
def load_sections(csv_path, total_pages):
    df = pd.read_csv(csv_path)
    sections = []

    for _, row in df.iterrows():
        if not to_bool(row["include_in_rag"]):
            continue

        sections.append({
            "title": row["Title"],
            "from_page": int(row["from_page"]) - 1,
            "to_page": int(row["to_page"]) - 1
        })

    return sections


# =============== TEXT ===============
def extract_text_blocks(doc, pages):
    elements = []

    for p in pages:
        page = doc[p]

        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue

            text_parts = []
            is_blue_text = False

            for line in block["lines"]:
                for span in line["spans"]:
                    span_text = span["text"]
                    color = span.get("color")

                    # STRICT: blue + keyword must be in same span
                    if is_blue(color) and (
                        "figure" in span_text.lower()
                        or "table" in span_text.lower()
                    ):
                        is_blue_text = True

                    text_parts.append(span_text)

            text = clean_text(" ".join(text_parts))
            
            # NOISE REMOVAL: Skip if text is entirely noise
            if is_noise(text):
                continue
            
            # NOISE REMOVAL: Remove noise patterns from text
            text = remove_noise(text)

            if text:
                elements.append({
                    "type": "text",
                    "page": p,
                    "bbox": block["bbox"],
                    "content": text,
                    "is_blue": is_blue_text
                })

    return elements


# =============== TABLE HELPERS ===============
def fix_table_structure(table):
    fixed = []
    last_row = None

    for row in table:
        if last_row:
            row = [
                cell if cell not in ["", None] else last_row[i]
                for i, cell in enumerate(row)
            ]
        fixed.append(row)
        last_row = row

    return fixed


def remove_repeated_headers(table):
    header = table[0]
    return [header] + [row for row in table[1:] if row != header]


def table_to_text(table):
    header = table[0]
    rows = table[1:]

    text = "HEADER: " + ", ".join(map(str, header)) + "\n"

    for row in rows:
        text += "ROW: " + ", ".join(map(str, row)) + "\n"

    return text.strip()


def summarize_table(table_text):
    lines = table_text.split("\n")
    return f"Table Summary:\n{lines[0]}\n{lines[1] if len(lines)>1 else ''}"


# =============== TABLE ===============
def extract_tables(pdf_path, pages):
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for p in pages:
            for table in pdf.pages[p].find_tables():
                data = table.extract()

                if not data:
                    continue

                data = fix_table_structure(data)
                data = remove_repeated_headers(data)

                tables.append({
                    "type": "table",
                    "page": p,
                    "bbox": table.bbox,
                    "content": data
                })

    return tables


# =============== IMAGE ===============
def extract_figure_images(doc, text_elements):
    images = []

    for e in text_elements:
        if get_caption_type(e["content"], e["is_blue"]) != "image":
            continue

        page = doc[e["page"]]
        rect = page.rect

        x0, y0, x1, y1 = e["bbox"]
        height = rect.height

        # extract ABOVE caption
        top = max(0, y0 - height * 0.5)
        bottom = y0 + height * 0.05

        clip = fitz.Rect(rect.x0, top, rect.x1, bottom)

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)

        path = os.path.join(
            IMAGE_DIR,
            f"page{e['page']}_figure_{len(images)}.png"
        )

        pix.save(path)

        images.append({
            "type": "image",
            "page": e["page"],
            "bbox": clip,
            "content": path,
            "caption": e["content"]
        })

    return images


# =============== MERGE ===============
def merge_elements(texts, tables, images):
    elements = texts + tables + images

    elements.sort(
        key=lambda x: (x["page"], x["bbox"][1])
    )

    return elements


# =============== RELATION ===============
def attach_relationships(elements):
    for i, e in enumerate(elements):

        if e["type"] == "text":
            if get_caption_type(e["content"], e.get("is_blue", False)) == "table":
                for j in range(i + 1, min(i + 5, len(elements))):
                    if elements[j]["type"] == "table":
                        elements[j]["caption"] = e["content"]
                        break

    return elements


# =============== TABLE MERGE ===============
def merge_split_tables(elements):
    tables = [e for e in elements if e["type"] == "table"]
    others = [e for e in elements if e["type"] != "table"]

    merged = []
    current = None

    for t in tables:
        if "caption" in t:
            if current:
                merged.append(current)
            current = t
        else:
            if current:
                current["content"].extend(t["content"])

    if current:
        merged.append(current)

    return others + merged


# =============== CHUNK WITH SLIDING WINDOW & TOKEN COUNTING ===============
def create_chunks(elements, section, max_tokens=MAX_TOKENS, overlap_pct=OVERLAP_PCT):
    """
    Create chunks using token-based sliding window with overlap.
    
    Args:
        elements (list): List of extracted elements
        section (dict): Section metadata (title, page range)
        max_tokens (int): Maximum tokens per chunk (default: 1000)
        overlap_pct (float): Overlap percentage (default: 0.15 = 15%)
    
    Returns:
        list: List of chunk dicts with text, tables, images, page info
    """
    chunks = []
    overlap_tokens = int(max_tokens * overlap_pct)
    
    # Separate text content for sliding window
    text_elements = [e for e in elements if e["type"] == "text"]
    tables = [e for e in elements if e["type"] == "table"]
    images = [e for e in elements if e["type"] == "image"]
    
    if not text_elements:
        return chunks
    
    # Track which tables/images have been assigned to chunks (to avoid duplication)
    used_table_indices = set()
    used_image_indices = set()
    
    # Build sliding window buffer
    buffer = ""
    buffer_start_page = text_elements[0]["page"]
    buffer_end_page = buffer_start_page
    buffer_token_count = 0
    
    element_idx = 0
    
    while element_idx < len(text_elements):
        e = text_elements[element_idx]
        text_content = e["content"]
        
        # Skip captions (they're only labels)
        if get_caption_type(text_content, e.get("is_blue", False)):
            element_idx += 1
            continue
        
        text_tokens = TokenCounter.count_tokens(text_content)
        
        # If adding this element would exceed max, emit chunk
        if buffer and (buffer_token_count + text_tokens > max_tokens):
            # Create chunk from current buffer
            chunk = create_chunk_from_buffer(
                buffer, 
                buffer_start_page, 
                buffer_end_page,
                tables,
                images,
                used_table_indices,
                used_image_indices,
                len(chunks)
            )
            chunks.append(chunk)
            
            # Create overlap: keep last OVERLAP_TOKENS
            # by truncating buffer to approximately overlap_tokens worth of text
            buffer = truncate_buffer_for_overlap(buffer, overlap_tokens)
            buffer_token_count = TokenCounter.count_tokens(buffer)
            
            # Add current element to overlapped buffer
            buffer += " " + text_content
            buffer_token_count += text_tokens
            buffer_end_page = e["page"]
        else:
            # Add to buffer
            if not buffer:
                buffer_start_page = e["page"]
            
            buffer += " " + text_content if buffer else text_content
            buffer_token_count += text_tokens
            buffer_end_page = e["page"]
        
        element_idx += 1
    
    # Emit final chunk if buffer has content
    if buffer and TokenCounter.count_tokens(buffer.strip()) > 0:
        chunk = create_chunk_from_buffer(
            buffer,
            buffer_start_page,
            buffer_end_page,
            tables,
            images,
            used_table_indices,
            used_image_indices,
            len(chunks)
        )
        chunks.append(chunk)
    
    return chunks


def truncate_buffer_for_overlap(buffer, target_overlap_tokens):
    """
    Truncate buffer to approximately target_overlap_tokens from the END.
    This creates the overlap between chunks.
    
    Args:
        buffer (str): Current buffer text
        target_overlap_tokens (int): Target number of overlap tokens
    
    Returns:
        str: Truncated buffer (overlap portion)
    """
    tokens = TokenCounter.encode(buffer)
    
    if len(tokens) <= target_overlap_tokens:
        return buffer
    
    # Keep last target_overlap_tokens
    overlap_token_ids = tokens[-target_overlap_tokens:]
    
    # Decode back to text
    encoder = TokenCounter.get_encoder()
    overlap_text = encoder.decode(overlap_token_ids)
    
    return overlap_text


def create_chunk_from_buffer(buffer_text, start_page, end_page, tables, images, used_table_indices, used_image_indices, chunk_id):
    """
    Create a chunk object from accumulated buffer text and associated elements.
    Only includes tables/images that haven't been used yet (no duplication).
    Adds references to nearby tables/images in other chunks for context continuity.
    
    Args:
        buffer_text (str): Accumulated text for this chunk
        start_page (int): Starting page number
        end_page (int): Ending page number
        tables (list): All table elements
        images (list): All image elements
        used_table_indices (set): Indices of tables already used in previous chunks
        used_image_indices (set): Indices of images already used in previous chunks
        chunk_id (int): Chunk sequence number
    
    Returns:
        dict: Chunk with text, tables, images, page info, and references
    """
    chunk_tables = []
    chunk_images = []
    table_references = []  # References to tables in nearby chunks
    image_references = []  # References to images in nearby chunks
    
    # Collect tables that fall within this chunk's page range AND haven't been used yet
    for idx, table in enumerate(tables):
        if idx not in used_table_indices and start_page <= table["page"] <= end_page:
            t = table_to_text(table["content"])
            t = summarize_table(t) + "\n\n" + t
            if "caption" in table:
                t = table["caption"] + "\n" + t
            chunk_tables.append(t)
            used_table_indices.add(idx)  # Mark as used
    
    # Collect images that fall within this chunk's page range AND haven't been used yet
    for idx, image in enumerate(images):
        if idx not in used_image_indices and start_page <= image["page"] <= end_page:
            img_text = f"{image['caption']}\n[Image: {os.path.basename(image['content'])}]"
            chunk_images.append(img_text)
            used_image_indices.add(idx)  # Mark as used
    
    # Add REFERENCES to nearby tables/images that are NOT in this chunk
    # but are close by (within ±2 pages) for context
    REFERENCE_PAGE_RANGE = 2  # Look ±2 pages for nearby elements
    
    for idx, table in enumerate(tables):
        # Skip if already used or in current chunk
        if idx in used_table_indices:
            continue
        
        # Include if nearby (within REFERENCE_PAGE_RANGE)
        if abs(table["page"] - start_page) <= REFERENCE_PAGE_RANGE or \
           abs(table["page"] - end_page) <= REFERENCE_PAGE_RANGE:
            caption = table.get("caption", "Unnamed table")
            page_num = table["page"] + 1  # Convert to 1-indexed for user readability
            table_references.append({
                "type": "table_reference",
                "caption": caption,
                "page": page_num,
                "table_idx": idx
            })
    
    for idx, image in enumerate(images):
        # Skip if already used or in current chunk
        if idx in used_image_indices:
            continue
        
        # Include if nearby (within REFERENCE_PAGE_RANGE)
        if abs(image["page"] - start_page) <= REFERENCE_PAGE_RANGE or \
           abs(image["page"] - end_page) <= REFERENCE_PAGE_RANGE:
            caption = image.get("caption", "Unnamed figure")
            page_num = image["page"] + 1  # Convert to 1-indexed for user readability
            image_references.append({
                "type": "image_reference",
                "caption": caption,
                "page": page_num,
                "image_idx": idx
            })
    
    return {
        "text": buffer_text.strip(),
        "tables": chunk_tables,
        "images": chunk_images,
        "table_references": table_references,  # ✅ References to tables in other chunks
        "image_references": image_references,  # ✅ References to images in other chunks
        "page_start": start_page,
        "page_end": end_page,
        "chunk_id": chunk_id,
        "token_count": TokenCounter.count_tokens(buffer_text)  # ✅ Token count metadata
    }


# =============== DOCUMENT ===============
def create_multimodal_documents(chunks, section):
    """
    Create LangChain Document objects from chunks.
    
    Args:
        chunks (list): List of chunk dicts
        section (dict): Section metadata
    
    Returns:
        list: List of Document objects
    """
    docs = []

    for chunk in chunks:
        # Build references section for text chunk
        references_text = ""
        
        # Add table references
        if chunk.get("table_references"):
            references_text += "\n--- Referenced Tables (see other chunks) ---\n"
            for ref in chunk["table_references"]:
                references_text += f"• {ref['caption']} (page {ref['page']})\n"
        
        # Add image references
        if chunk.get("image_references"):
            references_text += "\n--- Referenced Figures (see other chunks) ---\n"
            for ref in chunk["image_references"]:
                references_text += f"• {ref['caption']} (page {ref['page']})\n"
        
        # Main text chunk with references appended
        main_content = chunk["text"]
        if references_text:
            main_content += references_text
        
        docs.append(Document(
            page_content=main_content,
            metadata={
                "type": "text",
                "chunk_id": chunk["chunk_id"],
                "title": section["title"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "token_count": chunk["token_count"],  # ✅ Include token count in metadata
                "has_tables": len(chunk["tables"]) > 0,
                "has_images": len(chunk["images"]) > 0,
                "table_references": len(chunk.get("table_references", [])),
                "image_references": len(chunk.get("image_references", []))
            }
        ))

        # Table chunks
        for i, table_text in enumerate(chunk["tables"]):
            docs.append(Document(
                page_content=table_text,
                metadata={
                    "type": "table",
                    "chunk_id": chunk["chunk_id"],
                    "table_id": i,
                    "title": section["title"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"]
                }
            ))

        # Image chunks
        for i, image_text in enumerate(chunk["images"]):
            docs.append(Document(
                page_content=image_text,
                metadata={
                    "type": "image",
                    "chunk_id": chunk["chunk_id"],
                    "image_id": i,
                    "title": section["title"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"]
                }
            ))

    return docs


# =============== MAIN ===============
def process_pdf(pdf_path, noise_vars=None, max_tokens=MAX_TOKENS, overlap_pct=OVERLAP_PCT):
    """
    Process PDF and extract documents with token-based chunking and sliding window overlap.
    
    Args:
        pdf_path (str): Path to the PDF file
        noise_vars (list, optional): Custom noise variables
        max_tokens (int, optional): Maximum tokens per chunk
        overlap_pct (float, optional): Overlap percentage (0.0-1.0)
    
    Returns:
        list: List of Document objects
    """
    if noise_vars is None:
        noise_vars = NOISE_VARIABLES
    
    print(f"🔧 Initializing token counter with encoding: {ENCODING}")
    print(f"📊 Chunking config: max_tokens={max_tokens}, overlap={overlap_pct*100}% ({int(max_tokens*overlap_pct)} tokens)")
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    sections = load_sections(CSV_PATH, total_pages)
    all_docs = []
    total_chunks = 0

    for section in sections:
        print(f"\n📖 Processing: {section['title']}")

        pages = list(range(section["from_page"], section["to_page"] + 1))

        texts = extract_text_blocks(doc, pages)
        tables = extract_tables(pdf_path, pages)
        images = extract_figure_images(doc, texts)

        elements = merge_elements(texts, tables, images)
        elements = attach_relationships(elements)
        elements = merge_split_tables(elements)

        # ✅ Use updated create_chunks with token counting
        chunks = create_chunks(elements, section, max_tokens=max_tokens, overlap_pct=overlap_pct)
        docs = create_multimodal_documents(chunks, section)

        all_docs.extend(docs)
        total_chunks += len(chunks)
        
        print(f"   ✓ Created {len(chunks)} chunks → {len(docs)} documents")
        
        # Print chunk token stats for this section
        if chunks:
            chunk_tokens = [c["token_count"] for c in chunks]
            print(f"   Token stats: min={min(chunk_tokens)}, max={max(chunk_tokens)}, "
                  f"avg={sum(chunk_tokens)/len(chunk_tokens):.0f}")

    # Save output
    output_file = os.path.join(OUTPUT_DIR, "documents.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            [{"content": d.page_content, "metadata": d.metadata} for d in all_docs],
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\n✅ Saved {len(all_docs)} documents from {total_chunks} chunks")
    print(f"📁 Output: {output_file}\n")
    
    return all_docs


# =============== RUN ===============
if __name__ == "__main__":
    pdf_path = "C:\\rag\\data\\raw\\nasa_handbook.pdf"

    # Option 1: Use default settings (max_tokens=1000, overlap=15%)
    docs = process_pdf(pdf_path)
    
    # Option 2: Use custom settings (uncomment to use)
    # docs = process_pdf(
    #     pdf_path, 
    #     max_tokens=800,
    #     overlap_pct=0.20  # 20% overlap instead of 15%
    # )

    print(f"First document example:")
    print(f"  Content: {docs[0].page_content[:200]}...")
    print(f"  Metadata: {docs[0].metadata}")
    print(f"Total Documents: {len(docs)}")
