from __future__ import annotations

from typing import Any

import httpx
import streamlit as st

DEFAULT_API_BASE = "http://localhost:8000"


def render_source_documents(documents: list[dict[str, Any]]) -> None:
    """Show retrieved handbook locations (section titles, page ranges) below the answer."""
    if not documents:
        return
    with st.expander("References (retrieved sources)", expanded=False):
        for i, doc in enumerate(documents, 1):
            meta = doc.get("metadata") or {}
            title = (meta.get("parent_title") or meta.get("source") or "Source").strip()
            page = int(meta.get("page") or 0)
            page_end = int(meta.get("page_end") or 0)
            if page_end and page_end != page:
                pages = f"pages {page}–{page_end}"
            else:
                pages = f"page {page}" if page else "page unknown"
            st.markdown(f"**{i}.** {title} ({pages})")
            snippet = str(doc.get("page_content") or "").strip()
            if len(snippet) > 320:
                snippet = snippet[:320].rstrip() + "…"
            if snippet:
                st.caption(snippet)


def build_memory(messages: list[dict[str, Any]], keep_turns: int = 6) -> str | None:
    if not messages:
        return None

    turns = messages[-(keep_turns * 2) :]
    lines: list[str] = []
    for item in turns:
        role = item.get("role", "assistant")
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role.title()}: {content}")

    return "\n".join(lines) if lines else None


st.set_page_config(page_title="NASA RAG Frontend", page_icon="N", layout="wide")
st.title("NASA Systems Engineering RAG")

with st.sidebar:
    st.header("Connection")
    api_base = st.text_input("FastAPI base URL", value=DEFAULT_API_BASE).strip().rstrip("/")
    request_timeout_s = st.slider("Request timeout (s)", min_value=10, max_value=180, value=90, step=5)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_source_documents(message.get("documents") or [])

prompt = st.chat_input("Ask a question about the NASA handbook")
if prompt:
    user_content = prompt.strip()
    if not user_content:
        st.warning("Please enter a non-empty question.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_content})
        with st.chat_message("user"):
            st.markdown(user_content)

        with st.chat_message("assistant"):
            with st.spinner("Querying backend..."):
                try:
                    memory_str = build_memory(st.session_state.messages)
                    with httpx.Client(timeout=request_timeout_s) as client:
                        response = client.post(
                            f"{api_base}/api/query",
                            json={"query": user_content, "memory_str": memory_str},
                        )
                    response.raise_for_status()
                    data = response.json()
                    answer = str(data.get("answer", "")).strip() or "No answer returned."
                    documents = data.get("documents") or []

                    st.markdown(answer)
                    render_source_documents(documents if isinstance(documents, list) else [])
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "documents": documents if isinstance(documents, list) else [],
                        }
                    )
                except httpx.HTTPStatusError as exc:
                    details = exc.response.text[:3000]
                    message = f"Backend returned HTTP {exc.response.status_code}: {details}"
                    st.error(message)
                    st.session_state.messages.append({"role": "assistant", "content": message, "documents": []})
                except Exception as exc:
                    message = f"Request failed: {exc}"
                    st.error(message)
                    st.session_state.messages.append({"role": "assistant", "content": message, "documents": []})
