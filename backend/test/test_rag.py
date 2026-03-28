import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.retriever.rag_chain import RAGChain
from backend.retriever.context_builder import ContextBuilder


class FakeRetriever:
    def invoke(self, _query: str):
        return [
            Document(
                page_content="Phase A defines mission concepts and initial requirements.",
                metadata={
                    "source": "NASA Systems Engineering Handbook",
                    "page": 42,
                    "parent_id": "phase_a",
                    "parent_title": "Phase A: Concept",
                    "rrf_score": 0.032,
                },
            )
        ]


def test_lcel_chain_with_fake_dependencies() -> None:
    fake_model = RunnableLambda(lambda _prompt: AIMessage(content="mocked answer"))
    chain = RAGChain(retriever=FakeRetriever(), model=fake_model)

    prepared = chain.prepare("What does the systems engineering process flow look like?")
    docs = prepared["documents"]
    prompt_vars = prepared["prompt_vars"]

    assert len(docs) == 1
    assert prompt_vars["token_counts"]["total"] > 0
    assert prompt_vars["token_counts"]["total"] <= chain.context_builder.token_budget
    assert "Parent Sections" in (prompt_vars["context"] or "") or "Retrieved Documents" in (prompt_vars["context"] or "")

    answer = chain.invoke("What are the NASA project phases?")
    assert answer == "mocked answer"


def test_context_builder_clips_large_context_before_compression() -> None:
    builder = ContextBuilder()
    large_text = "systems engineering " * 1200
    documents = [
        Document(
            page_content=large_text,
            metadata={
                "source": f"Section {index}",
                "page": 10 + index,
                "page_end": 11 + index,
                "parent_id": f"parent_{index}",
                "parent_title": f"Parent {index}",
                "rrf_score": 0.05 - (index * 0.001),
                "entry_type": "document",
            },
        )
        for index in range(5)
    ]

    prompt_vars = builder.build_prompt_vars(
        documents=documents,
        query="What does the systems engineering process flow look like?",
    )

    assert prompt_vars["compression_needed"] is False
    assert prompt_vars["token_counts"]["context"] <= builder.max_context_tokens
    assert "Relevant parent sections:" in prompt_vars["context"]
    assert "[4]" not in prompt_vars["context"]
    assert "...[truncated]" in prompt_vars["context"]