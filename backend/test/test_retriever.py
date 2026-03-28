# Test file: test_retriever.py

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.retriever.retriever import HybridRetriever

# Initialize
retriever = HybridRetriever()

# Test query
query = "What are the NASA project phases?"
docs = retriever.invoke(query)

print(f"Retrieved {len(docs)} documents")
for doc in docs[:3]:
    print(f"- {doc.metadata['source']} (RRF: {doc.metadata['rrf_score']:.4f})")
    print(f"  {doc.page_content[:100]}...\n")