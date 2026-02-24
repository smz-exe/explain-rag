"""Streamlit admin dashboard for ExplainRAG."""

import os
import re
from datetime import datetime

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
    """Render the sidebar with health status and paper ingestion."""
    st.sidebar.title("Admin Actions")

    # Health check
    client = get_client()
    try:
        stats = client.get_stats()
        if stats.backend_status == "healthy":
            st.sidebar.success(f"Backend: {stats.backend_status}")
        else:
            st.sidebar.warning(f"Backend: {stats.backend_status}")
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


def render_system_metrics():
    """Render the system metrics cards."""
    st.subheader("System Metrics")

    client = get_client()
    try:
        stats = client.get_stats()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Papers", stats.papers_count)
        with col2:
            st.metric("Chunks", stats.chunks_count)
        with col3:
            st.metric("Queries", stats.queries_count)
        with col4:
            if stats.backend_status == "healthy":
                st.metric("Status", "Healthy")
            else:
                st.metric("Status", stats.backend_status)
    except Exception as e:
        st.error(f"Failed to load stats: {e}")


def render_papers_table():
    """Render the papers management table."""
    st.subheader("Papers")

    client = get_client()
    try:
        papers = client.list_papers()

        if not papers:
            st.info("No papers ingested yet. Use the sidebar to add papers.")
            return

        # Create table data
        for paper in papers:
            col1, col2, col3, col4 = st.columns([4, 2, 1, 1])

            with col1:
                st.write(f"**{paper.title[:60]}{'...' if len(paper.title) > 60 else ''}**")
            with col2:
                st.write(f"`{paper.arxiv_id}`")
            with col3:
                st.write(f"{paper.chunk_count} chunks")
            with col4:
                # Delete button with confirmation
                confirm_key = f"confirm_{paper.paper_id}"

                if st.session_state.get(confirm_key, False):
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes", key=f"yes_{paper.paper_id}", type="primary"):
                            try:
                                result = client.delete_paper(paper.paper_id)
                                st.success(f"Deleted {result.get('deleted_chunks', 0)} chunks")
                                st.session_state[confirm_key] = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    with col_no:
                        if st.button("No", key=f"no_{paper.paper_id}"):
                            st.session_state[confirm_key] = False
                            st.rerun()
                else:
                    if st.button("Delete", key=f"delete_{paper.paper_id}", type="secondary"):
                        st.session_state[confirm_key] = True
                        st.rerun()

            st.divider()

    except Exception as e:
        st.error(f"Failed to load papers: {e}")


def render_queries_table():
    """Render the recent queries table."""
    st.subheader("Recent Queries")

    client = get_client()
    try:
        queries = client.list_queries(limit=10)

        if not queries:
            st.info("No queries yet. Use the Next.js frontend or test query section below.")
            return

        for query in queries:
            col1, col2 = st.columns([5, 1])

            with col1:
                # Truncate question for display
                question_preview = query.question[:80] + "..." if len(query.question) > 80 else query.question
                st.write(f"**{question_preview}**")
                st.caption(f"Answer: {query.answer_preview[:100]}...")

            with col2:
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(query.created_at.replace("Z", "+00:00"))
                    time_str = dt.strftime("%m/%d %H:%M")
                except Exception:
                    time_str = query.created_at[:16]
                st.caption(time_str)

            st.divider()

    except Exception as e:
        st.error(f"Failed to load queries: {e}")


def highlight_citations(text: str) -> str:
    """Add HTML styling to citation markers."""

    def replace_citation(match):
        num = match.group(1)
        return f'<span style="background-color: #3b82f6; color: white; padding: 0 4px; border-radius: 4px; font-size: 0.8em;">[{num}]</span>'

    return re.sub(r"\[(\d+)\]", replace_citation, text)


def render_test_query():
    """Render the test query section (collapsible)."""
    with st.expander("Test Query (Debug)", expanded=False):
        st.caption("Quick query interface for testing and debugging.")

        question = st.text_area(
            "Question",
            placeholder="e.g., What is self-attention?",
            height=80,
            key="test_query_input",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            top_k = st.slider("Top-K", min_value=1, max_value=20, value=5, key="test_top_k")
        with col2:
            enable_reranking = st.checkbox("Reranking", value=False, key="test_reranking")
        with col3:
            pass  # Empty column for spacing

        if st.button("Run Test Query", disabled=not question, key="test_submit"):
            client = get_client()
            with st.status("Processing...", expanded=True) as status:
                try:
                    response = client.query(
                        question=question,
                        top_k=top_k,
                        enable_reranking=enable_reranking,
                    )
                    st.session_state.test_response = response
                    status.update(label="Done!", state="complete")
                except Exception as e:
                    status.update(label="Failed", state="error")
                    st.error(f"Error: {e}")

        # Display test results
        if "test_response" in st.session_state:
            response = st.session_state.test_response

            # Answer
            st.markdown("**Answer:**")
            answer = response.get("answer", "")
            st.markdown(highlight_citations(answer), unsafe_allow_html=True)

            # Faithfulness score
            score = response.get("faithfulness", {}).get("score", 0)
            st.markdown(f"**Faithfulness:** {score:.0%}")

            # Timing
            trace = response.get("trace", {})
            st.caption(f"Total time: {trace.get('total_time_ms', 0):.0f}ms")


def main():
    """Main entry point for the Streamlit admin dashboard."""
    st.set_page_config(
        page_title="ExplainRAG Admin",
        page_icon="🔧",
        layout="wide",
    )

    st.title("ExplainRAG Admin Dashboard")
    st.caption("System monitoring and paper management")

    render_sidebar()

    # Main content area
    render_system_metrics()

    st.divider()

    # Two columns for papers and queries
    col1, col2 = st.columns(2)
    with col1:
        render_papers_table()
    with col2:
        render_queries_table()

    st.divider()

    # Test query section at the bottom
    render_test_query()


if __name__ == "__main__":
    main()
