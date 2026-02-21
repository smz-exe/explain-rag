"""Streamlit dashboard for ExplainRAG."""

import os
import re

import streamlit as st

from src.adapters.inbound.streamlit.api_client import APIClient

# Configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")


def get_client() -> APIClient:
    """Get or create the API client."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient(base_url=FASTAPI_URL)
    return st.session_state.api_client


def render_sidebar():
    """Render the sidebar with paper management."""
    st.sidebar.title("Paper Management")

    # Health check
    client = get_client()
    try:
        health = client.health_check()
        if health.status == "healthy":
            st.sidebar.success(f"Backend: {health.status}")
            st.sidebar.caption(
                f"Chunks: {health.stats.get('total_chunks', 0)} | "
                f"Papers: {health.stats.get('total_papers', 0)}"
            )
        else:
            st.sidebar.warning(f"Backend: {health.status}")
    except Exception as e:
        st.sidebar.error(f"Backend unavailable: {e}")
        return

    st.sidebar.divider()

    # Add paper section
    st.sidebar.subheader("Add Paper")
    arxiv_id = st.sidebar.text_input(
        "arXiv ID",
        placeholder="e.g., 1706.03762",
        help="Enter an arXiv paper ID to ingest",
    )
    if st.sidebar.button("Ingest Paper", disabled=not arxiv_id):
        with st.sidebar.status("Ingesting paper..."):
            try:
                result = client.ingest_papers([arxiv_id])
                if result.success:
                    st.sidebar.success(f"Ingested: {', '.join(result.success)}")
                    st.rerun()
                if result.failed:
                    for fail in result.failed:
                        st.sidebar.error(f"Failed: {fail}")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    st.sidebar.divider()

    # List papers
    st.sidebar.subheader("Ingested Papers")
    try:
        papers = client.list_papers()
        if papers:
            for paper in papers:
                with st.sidebar.expander(f"{paper.title[:40]}...", expanded=False):
                    st.caption(f"arXiv: {paper.arxiv_id}")
                    st.caption(f"Chunks: {paper.chunk_count}")

                    # Delete button with confirmation
                    delete_key = f"delete_{paper.paper_id}"
                    confirm_key = f"confirm_{paper.paper_id}"

                    if st.session_state.get(confirm_key, False):
                        st.warning("Are you sure?")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Yes", key=f"yes_{paper.paper_id}"):
                                try:
                                    result = client.delete_paper(paper.paper_id)
                                    st.success(f"Deleted {result.get('deleted_chunks', 0)} chunks")
                                    st.session_state[confirm_key] = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        with col2:
                            if st.button("No", key=f"no_{paper.paper_id}"):
                                st.session_state[confirm_key] = False
                                st.rerun()
                    else:
                        if st.button("Delete", key=delete_key, type="secondary"):
                            st.session_state[confirm_key] = True
                            st.rerun()
        else:
            st.sidebar.info("No papers ingested yet")
    except Exception as e:
        st.sidebar.error(f"Failed to load papers: {e}")


def render_query_interface():
    """Render the main query interface."""
    st.title("ExplainRAG")
    st.caption("Explainable Retrieval-Augmented Generation for Academic Papers")

    # Query input
    question = st.text_area(
        "Ask a question",
        placeholder="e.g., What is self-attention and how does it work?",
        height=100,
    )

    # Options in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Top-K chunks", min_value=1, max_value=50, value=10)
    with col2:
        enable_reranking = st.checkbox("Enable reranking", value=False)
    with col3:
        # Paper filter (optional)
        client = get_client()
        try:
            papers = client.list_papers()
            paper_options = {p.title[:50]: p.paper_id for p in papers}
            selected_papers = st.multiselect(
                "Filter by papers",
                options=list(paper_options.keys()),
                help="Leave empty to search all papers",
            )
            paper_ids = [paper_options[p] for p in selected_papers] if selected_papers else None
        except Exception:
            paper_ids = None

    # Submit button
    if st.button("Submit Query", type="primary", disabled=not question):
        with st.status("Processing query...", expanded=True) as status:
            try:
                st.write("Embedding query...")
                st.write("Searching vector store...")
                if enable_reranking:
                    st.write("Reranking chunks...")
                st.write("Generating answer...")
                st.write("Verifying faithfulness...")

                response = client.query(
                    question=question,
                    top_k=top_k,
                    paper_ids=paper_ids,
                    enable_reranking=enable_reranking,
                )
                st.session_state.last_response = response
                status.update(label="Query completed!", state="complete")
            except Exception as e:
                status.update(label="Query failed", state="error")
                st.error(f"Error: {e}")
                return

    # Display results
    if "last_response" in st.session_state:
        render_results(st.session_state.last_response)


def highlight_citations(text: str) -> str:
    """Add HTML styling to citation markers."""
    # Replace [1], [2], etc. with styled spans
    def replace_citation(match):
        num = match.group(1)
        return f'<span style="background-color: #3b82f6; color: white; padding: 0 4px; border-radius: 4px; font-size: 0.8em;">[{num}]</span>'

    return re.sub(r"\[(\d+)\]", replace_citation, text)


def render_results(response: dict):
    """Render the query results."""
    st.divider()

    # Answer with faithfulness badge
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("Answer")
    with col2:
        score = response.get("faithfulness", {}).get("score", 0)
        if score >= 0.8:
            st.success(f"Faithfulness: {score:.0%}")
        elif score >= 0.5:
            st.warning(f"Faithfulness: {score:.0%}")
        else:
            st.error(f"Faithfulness: {score:.0%}")

    # Answer text with highlighted citations
    answer = response.get("answer", "")
    st.markdown(highlight_citations(answer), unsafe_allow_html=True)

    # Two columns: chunks and faithfulness
    col1, col2 = st.columns(2)

    with col1:
        render_chunks_panel(response)

    with col2:
        render_faithfulness_report(response)

    # Timing trace
    render_timing_trace(response)


def render_chunks_panel(response: dict):
    """Render the retrieved chunks panel."""
    st.subheader("Retrieved Chunks")

    chunks = response.get("retrieved_chunks", [])
    if not chunks:
        st.info("No chunks retrieved")
        return

    for chunk in chunks:
        rank = chunk.get("rank", "?")
        similarity = chunk.get("similarity_score", 0)
        rerank = chunk.get("rerank_score")
        paper_title = chunk.get("paper_title", "Unknown")
        content = chunk.get("content", "")

        # Score display
        if rerank is not None:
            score_text = f"Sim: {similarity:.2f} | Rerank: {rerank:.2f}"
        else:
            score_text = f"Similarity: {similarity:.2f}"

        with st.expander(f"**[{rank}]** {paper_title[:30]}... ({score_text})"):
            st.caption(f"Chunk ID: {chunk.get('chunk_id', 'N/A')}")
            st.write(content)


def render_faithfulness_report(response: dict):
    """Render the faithfulness verification report."""
    st.subheader("Faithfulness Report")

    faithfulness = response.get("faithfulness", {})
    claims = faithfulness.get("claims", [])

    if not claims:
        st.info("No claims to verify")
        return

    for claim_data in claims:
        claim = claim_data.get("claim", "")
        verdict = claim_data.get("verdict", "unknown")
        reasoning = claim_data.get("reasoning", "")
        evidence = claim_data.get("evidence_chunk_ids", [])

        # Verdict badge
        if verdict == "supported":
            badge = ":green[Supported]"
        elif verdict == "partial":
            badge = ":orange[Partial]"
        else:
            badge = ":red[Unsupported]"

        with st.expander(f"{badge} {claim[:50]}..."):
            st.write(f"**Verdict:** {verdict}")
            st.write(f"**Reasoning:** {reasoning}")
            if evidence:
                st.write(f"**Evidence chunks:** {', '.join(evidence)}")


def render_timing_trace(response: dict):
    """Render the timing trace section."""
    with st.expander("Timing Trace"):
        trace = response.get("trace", {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Embedding", f"{trace.get('embedding_time_ms', 0):.0f}ms")
            st.metric("Retrieval", f"{trace.get('retrieval_time_ms', 0):.0f}ms")
        with col2:
            rerank_time = trace.get("reranking_time_ms")
            if rerank_time is not None:
                st.metric("Reranking", f"{rerank_time:.0f}ms")
            else:
                st.metric("Reranking", "N/A")
            st.metric("Generation", f"{trace.get('generation_time_ms', 0):.0f}ms")
        with col3:
            st.metric("Faithfulness", f"{trace.get('faithfulness_time_ms', 0):.0f}ms")
            st.metric("Total", f"{trace.get('total_time_ms', 0):.0f}ms")


def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="ExplainRAG",
        page_icon="📚",
        layout="wide",
    )

    render_sidebar()
    render_query_interface()


if __name__ == "__main__":
    main()
