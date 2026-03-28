"""CLI entry point for ad-hoc RAG queries (run from repo root: uv run python -m backend.cli)."""

import argparse

from backend.retriever.rag_chain import RAGChain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NASA handbook RAG CLI")
    parser.add_argument("query", nargs="+", help="Question to ask the assistant")
    parser.add_argument("--memory", default=None, help="Optional conversation memory text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = " ".join(args.query).strip()

    chain = RAGChain()
    answer = chain.invoke(query, memory_str=args.memory)
    print(answer)


if __name__ == "__main__":
    main()
