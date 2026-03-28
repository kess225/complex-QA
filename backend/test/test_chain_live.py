import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.retriever.rag_chain import RAGChain


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_OLLAMA") != "1",
    reason="Set RUN_LIVE_OLLAMA=1 to run live Ollama integration tests.",
)
def test_lcel_chain_live_ollama() -> None:
    chain = RAGChain()
    answer = chain.invoke("What are the NASA project phases?")

    assert isinstance(answer, str)
    assert answer.strip() != ""
