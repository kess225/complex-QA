"""
rag_chain.py
------------
LangChain LCEL composition for end-to-end RAG:
query -> HybridRetriever -> ContextBuilder -> ChatPromptTemplate -> ChatOllama -> string output.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from backend.retriever.context_builder import ContextBuilder
from backend.retriever.retriever import HybridRetriever

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = BACKEND_ROOT / "settings.yaml"

if SETTINGS_FILE.exists():
    with open(SETTINGS_FILE, encoding="utf-8") as handle:
        CFG = yaml.safe_load(handle) or {}
else:
    CFG = {}

OLLAMA_CFG = CFG.get("ollama", {})
MODEL_CFG = CFG.get("models", {})
GROQ_CFG = CFG.get("groq", {})

LLM_CFG = MODEL_CFG.get("llm", {})
LLM_PROVIDER = str(LLM_CFG.get("provider", "ollama")).strip().lower()
OLLAMA_MODEL = LLM_CFG.get("standard", "qwen2.5:7b")
OLLAMA_BASE = OLLAMA_CFG.get("base_url", "http://localhost:11434")
OLLAMA_TIMEOUT = OLLAMA_CFG.get("timeout", 60)

GROQ_PROFILE = LLM_CFG.get("groq_profile", "llama3-fast")
_groq_profile_cfg = GROQ_CFG.get("providers", {}).get(GROQ_PROFILE, {})
GROQ_MODEL = _groq_profile_cfg.get("model", "llama-3.1-8b-instant")
GROQ_TEMPERATURE = float(_groq_profile_cfg.get("temperature", 0.1))
GROQ_MAX_TOKENS = int(_groq_profile_cfg.get("max_tokens", 1024))
GROQ_API_KEY_ENV = GROQ_CFG.get("api_key_env", "GROQ_API_KEY")


def _create_ollama_model() -> Runnable[Any, AIMessage]:
    try:
        ollama_mod = importlib.import_module("langchain_ollama")
        chat_ollama_cls = getattr(ollama_mod, "ChatOllama")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "langchain_ollama is required for Ollama LLM calls. "
            "Install dependencies from pyproject.toml before running the RAG chain."
        ) from exc

    return chat_ollama_cls(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE,
        timeout=OLLAMA_TIMEOUT,
    )


def _create_groq_model() -> Runnable[Any, AIMessage]:
    import os

    api_key = os.getenv(GROQ_API_KEY_ENV, "").strip()
    if not api_key:
        raise EnvironmentError(
            f"Groq API key not set. Export {GROQ_API_KEY_ENV}=<your-key> before running."
        )
    try:
        groq_mod = importlib.import_module("langchain_groq")
        chat_groq_cls = getattr(groq_mod, "ChatGroq")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "langchain_groq is required for Groq LLM calls. "
            "Run: uv pip install langchain-groq"
        ) from exc

    return chat_groq_cls(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
    )


def _create_default_model() -> Runnable[Any, AIMessage]:
    if LLM_PROVIDER == "groq":
        return _create_groq_model()
    return _create_ollama_model()


class RAGChain:
    """LCEL-composed retrieval + generation chain."""

    def __init__(
        self,
        retriever: Any | None = None,
        context_builder: ContextBuilder | None = None,
        model: Runnable[Any, AIMessage] | None = None,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.context_builder = context_builder or ContextBuilder()
        self.model = model or _create_default_model()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("human", "{human_input}"),
            ]
        )
        self._chain = self._build_chain()

    def _build_chain(self):
        return (
            RunnablePassthrough.assign(documents=RunnableLambda(self._retrieve_documents))
            | RunnableLambda(self._to_prompt_input)
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def _retrieve_documents(self, payload: dict[str, Any]) -> list[Document]:
        query = str(payload.get("query") or "")
        return self.retriever.invoke(query)

    def _to_prompt_input(self, payload: dict[str, Any]) -> dict[str, str]:
        query = str(payload.get("query") or "")
        memory_str = payload.get("memory_str")
        documents = payload.get("documents") or []

        prompt_vars = self.context_builder.build_prompt_vars(
            documents=documents,
            query=query,
            memory_str=memory_str,
        )

        human_parts: list[str] = []
        if prompt_vars.get("memory_block"):
            human_parts.append(str(prompt_vars["memory_block"]))
        if prompt_vars.get("context_block"):
            human_parts.append(str(prompt_vars["context_block"]))
        human_parts.append(f"User Query:\n{query}")

        return {
            "system_prompt": str(prompt_vars["system_prompt"]),
            "human_input": "\n\n".join(human_parts),
        }

    def prepare(self, query: str, memory_str: str | None = None) -> dict[str, Any]:
        """Return retrieval output and prompt variables for debugging/tests."""
        documents = self.retriever.invoke(query)
        prompt_vars = self.context_builder.build_prompt_vars(
            documents=documents,
            query=query,
            memory_str=memory_str,
        )
        return {
            "documents": documents,
            "prompt_vars": prompt_vars,
        }

    def invoke(self, query: str, memory_str: str | None = None) -> str:
        return self._chain.invoke({"query": query, "memory_str": memory_str})

    def batch(self, queries: list[str], memory_str: str | None = None) -> list[str]:
        payloads = [{"query": q, "memory_str": memory_str} for q in queries]
        return self._chain.batch(payloads)
