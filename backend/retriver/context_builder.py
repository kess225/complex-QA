"""
context_builder.py
-----------------
Assembles retrieval results + conversation memory into a contextualized prompt.

Your structure:
  rag/
  └── retriever/
      ├── retriever.py
      └── context_builder.py  ← This file

Usage:
    from rag.retriever.context_builder import ContextBuilder
    from rag.retriever.retriever import HybridRetriever
    
    retriever = HybridRetriever()
    builder = ContextBuilder()
    
    docs = retriever.invoke("your question")
    result = builder.build(documents=docs, query="your question")
    prompt = result["prompt"]
"""

import logging
from pathlib import Path
from typing import Optional

import tiktoken
import yaml

log = logging.getLogger(__name__)

# ====================================================================
# CONFIG
# ====================================================================
BACKEND_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = BACKEND_ROOT / "settings.yaml"

# Load settings if exists
if SETTINGS_FILE.exists():
    with open(SETTINGS_FILE, encoding="utf-8") as f:
        CFG = yaml.safe_load(f) or {}
else:
    CFG = {}

CONTEXT_CFG = CFG.get("context_builder", {})
ENCODING = "cl100k_base"  # GPT/Claude standard
TOKEN_BUDGET = int(CONTEXT_CFG.get("token_budget", 3800))
MAX_CONTEXT_TOKENS = int(CONTEXT_CFG.get("max_context_tokens", 2000))
MAX_DOCS = int(CONTEXT_CFG.get("max_docs", 3))
MAX_DOC_TOKENS = int(CONTEXT_CFG.get("max_doc_tokens", 350))
MAX_PARENT_SECTIONS = int(CONTEXT_CFG.get("max_parent_sections", 3))
MAX_PARENT_SECTION_TOKENS = int(CONTEXT_CFG.get("max_parent_section_tokens", 250))


# ====================================================================
# TOKEN COUNTER
# ====================================================================

class TokenCounter:
    """Efficient token counting using tiktoken."""
    _encoder = None
    
    @classmethod
    def get_encoder(cls):
        if cls._encoder is None:
            cls._encoder = tiktoken.get_encoding(ENCODING)
        return cls._encoder
    
    @classmethod
    def count(cls, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        encoder = cls.get_encoder()
        return len(encoder.encode(text))


# ====================================================================
# CONTEXT BUILDER
# ====================================================================

class ContextBuilder:
    """
    Assembles prompt with token budgeting and memory integration.
    
    Budget breakdown (3800 tokens):
    - System prompt:  ~150-200 tokens
    - Memory:         Variable (0-1000)
    - Context:        Variable (max 2000)
    - Query:          ~30-80 tokens
    
    If context > MAX_CONTEXT_TOKENS (2000), triggers Summariser Path B
    (compression flag is set for downstream implementation).
    """
    
    def __init__(self, token_budget: int = TOKEN_BUDGET):
        self.token_budget = token_budget
        self.max_context_tokens = MAX_CONTEXT_TOKENS
        self.max_docs = MAX_DOCS
        self.max_doc_tokens = MAX_DOC_TOKENS
        self.max_parent_sections = MAX_PARENT_SECTIONS
        self.max_parent_section_tokens = MAX_PARENT_SECTION_TOKENS
    
    def _get_system_prompt(self) -> str:
        """Return the system prompt."""
        return """You are a helpful AI assistant specializing in NASA systems engineering.
        
Your role is to:
1. Answer questions about NASA projects, processes, and systems engineering practices
2. Cite specific sections and pages when referencing the handbook
3. Provide clear, technical explanations
4. Acknowledge limitations or areas outside your knowledge

When referencing documents, always include the source section and page numbers."""
    
    def _clip_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """Clip text to a token limit while preserving valid decoding."""
        if not text or max_tokens <= 0:
            return ""

        encoder = TokenCounter.get_encoder()
        tokens = encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text

        clipped = encoder.decode(tokens[:max_tokens]).rstrip()
        return f"{clipped}\n...[truncated]"

    def _format_page_range(self, page_start, page_end) -> str:
        """Format page metadata for compact source labels."""
        if page_start in (None, "", "?") and page_end in (None, "", "?"):
            return ""
        if page_end in (None, "", "?", page_start):
            return f"pages {page_start}"
        return f"pages {page_start}-{page_end}"

    def _get_parent_context(self, documents) -> str:
        """
        Extract unique parent sections from retrieved documents.
        Groups documents by parent_id for context richness.
        """
        parent_sections = {}
        fallback_summaries = {}

        for doc in documents:
            metadata = doc.metadata or {}
            entry_type = metadata.get("entry_type")
            parent_id = metadata.get("parent_id") or metadata.get("chunk_id")
            parent_title = metadata.get("parent_title") or metadata.get("source") or "Parent Section"

            if not parent_id:
                continue

            if entry_type == "parent":
                if parent_id in parent_sections:
                    continue
                parent_sections[parent_id] = {
                    "title": parent_title,
                    "content": self._clip_text_to_tokens(doc.page_content, self.max_parent_section_tokens),
                }
                continue

            if parent_id in fallback_summaries:
                continue

            page_label = self._format_page_range(metadata.get("page"), metadata.get("page_end"))
            fallback_summaries[parent_id] = f"- {parent_title}" + (f" ({page_label})" if page_label else "")

        context_parts = []
        for parent_id, section in list(parent_sections.items())[: self.max_parent_sections]:
            context_parts.append(f"## {section['title']}\n{section['content']}")

        if context_parts:
            return "\n\n".join(context_parts)

        summary_lines = list(fallback_summaries.values())[: self.max_parent_sections]
        if not summary_lines:
            return ""
        return "Relevant parent sections:\n" + "\n".join(summary_lines)

    def _format_documents(self, documents) -> str:
        """Format retrieved documents for context."""
        if not documents:
            return ""

        non_parent_documents = [
            doc for doc in documents if (doc.metadata or {}).get("entry_type") != "parent"
        ]
        selected_documents = non_parent_documents or list(documents)

        parts = []
        for i, doc in enumerate(selected_documents[: self.max_docs], 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            page_end = doc.metadata.get("page_end")
            rrf_score = doc.metadata.get("rrf_score", 0)
            page_label = self._format_page_range(page, page_end) or f"page {page}"
            content = self._clip_text_to_tokens(doc.page_content, self.max_doc_tokens)

            parts.append(
                f"[{i}] {source} ({page_label}, relevance: {rrf_score:.3f})\n{content}"
            )

        return "\n\n".join(parts)
    
    def build_prompt_vars(
        self,
        documents,
        query: str,
        memory_str: Optional[str] = None,
        memory_tokens: int = 0,
    ) -> dict:
        """
        Build final prompt with token accounting.
        
        Args:
            documents: List of LangChain Documents from retriever
            query: The user's question
            memory_str: Optional conversation history
            memory_tokens: Pre-calculated token count for memory
        
        Returns structured values suitable for LCEL prompt templates.
        """
        # Step 1: Count system prompt
        system_prompt = self._get_system_prompt()
        system_tokens = TokenCounter.count(system_prompt)
        log.debug(f"System prompt: {system_tokens} tokens")
        
        # Step 2: Account for memory
        if memory_tokens == 0 and memory_str:
            memory_tokens = TokenCounter.count(memory_str)
        log.debug(f"Memory: {memory_tokens} tokens")
        
        # Step 3: Build context from documents
        parent_context = self._get_parent_context(documents)
        doc_context = self._format_documents(documents)
        
        if parent_context and doc_context:
            context_str = f"## Parent Sections\n{parent_context}\n\n## Retrieved Documents\n{doc_context}"
        elif doc_context:
            context_str = f"## Retrieved Documents\n{doc_context}"
        else:
            context_str = ""
        
        context_tokens = TokenCounter.count(context_str)
        log.debug(f"Context (initial): {context_tokens} tokens")
        
        # Step 4: Count query
        query_tokens = TokenCounter.count(query)
        log.debug(f"Query: {query_tokens} tokens")
        
        # Step 5: Check if compression is needed (Path B trigger)
        compression_needed = context_tokens > self.max_context_tokens
        compressed_context = None
        
        if compression_needed:
            log.warning(
                f"Context {context_tokens} tokens > {self.max_context_tokens} threshold. "
                f"Compression needed (Path B)."
            )
            # In production, call Summariser Path B here
            # For now, just truncate to budget
            available_tokens = (
                self.token_budget
                - system_tokens
                - memory_tokens
                - query_tokens
                - 50  # Safety margin
            )
            
            if available_tokens < 500:
                log.warning(f"Very limited context budget: {available_tokens} tokens")
            
            # Truncate context to available tokens
            compressed_context = self._truncate_to_tokens(context_str, available_tokens)
            context_tokens = TokenCounter.count(compressed_context)
            log.info(f"Context (compressed): {context_tokens} tokens")
        
        final_context = compressed_context or context_str
        memory_block = ""
        if memory_str:
            memory_block = f"## Conversation Memory\n{memory_str}"

        context_block = ""
        if final_context:
            context_block = f"## Context\n{final_context}"
        
        # Calculate final token count
        total_tokens = (
            system_tokens
            + memory_tokens
            + context_tokens
            + query_tokens
        )
        
        log.info(
            f"Prompt assembled: "
            f"system={system_tokens}, memory={memory_tokens}, "
            f"context={context_tokens}, query={query_tokens}, "
            f"total={total_tokens}/{self.token_budget}"
        )
        
        return {
            "system_prompt": system_prompt,
            "memory": memory_str or "",
            "context": final_context,
            "query": query,
            "memory_block": memory_block,
            "context_block": context_block,
            "token_counts": {
                "system": system_tokens,
                "memory": memory_tokens,
                "context": context_tokens,
                "query": query_tokens,
                "total": total_tokens,
            },
            "compression_needed": compression_needed,
            "compressed_context": compressed_context,
        }

    def build(
        self,
        documents,
        query: str,
        memory_str: Optional[str] = None,
        memory_tokens: int = 0,
    ) -> dict:
        """
        Backwards-compatible prompt assembly helper.

        This method keeps the previous behavior (returning a full prompt string),
        while build_prompt_vars() exposes structured values for LCEL chains.
        """
        prompt_vars = self.build_prompt_vars(
            documents=documents,
            query=query,
            memory_str=memory_str,
            memory_tokens=memory_tokens,
        )

        prompt_parts = [prompt_vars["system_prompt"]]

        if prompt_vars["memory_block"]:
            prompt_parts.extend(["", prompt_vars["memory_block"]])

        if prompt_vars["context_block"]:
            prompt_parts.extend(["", prompt_vars["context_block"]])

        prompt_parts.extend(["", f"User Query:\n{query}"])
        final_prompt = "\n\n".join(prompt_parts)

        return {
            "prompt": final_prompt,
            "token_counts": prompt_vars["token_counts"],
            "compression_needed": prompt_vars["compression_needed"],
            "compressed_context": prompt_vars["compressed_context"],
        }
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        encoder = TokenCounter.get_encoder()
        tokens = encoder.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        return encoder.decode(truncated_tokens)


# ====================================================================
# USAGE EXAMPLE
# ====================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from langchain_core.documents import Document
    
    # Mock documents (from HybridRetriever)
    mock_docs = [
        Document(
            page_content="Phase A is the concept phase where requirements are defined.",
            metadata={
                "source": "NASA Systems Engineering Handbook",
                "page": 42,
                "parent_id": "phase_a",
                "parent_title": "Phase A: Concept",
                "rrf_score": 0.035,
            }
        ),
        Document(
            page_content="Phase B involves preliminary design activities.",
            metadata={
                "source": "NASA Systems Engineering Handbook",
                "page": 48,
                "parent_id": "phase_b",
                "parent_title": "Phase B: Preliminary Design",
                "rrf_score": 0.032,
            }
        ),
    ]
    
    # Build context
    builder = ContextBuilder()
    result = builder.build(
        documents=mock_docs,
        query="What are the NASA project phases?",
        memory_str="Previously discussed: basic concepts",
        memory_tokens=50,
    )
    
    print("=" * 60)
    print("ASSEMBLED PROMPT")
    print("=" * 60)
    print(result["prompt"])
    print("\n" + "=" * 60)
    print("TOKEN COUNTS")
    print("=" * 60)
    for key, value in result["token_counts"].items():
        print(f"  {key:12} : {value:5} tokens")
    print(f"  {'Budget':12} : {builder.token_budget:5} tokens")
    print(f"  {'Available':12} : {builder.token_budget - result['token_counts']['total']:5} tokens")
    
    if result["compression_needed"]:
        print("\n⚠️  Compression was needed (Path B triggered)")
