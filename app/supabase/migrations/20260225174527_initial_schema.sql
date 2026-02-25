-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Papers table: stores academic paper metadata
CREATE TABLE papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors TEXT[] NOT NULL DEFAULT '{}',
    abstract TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL,
    pdf_url TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index on arxiv_id for fast lookups
CREATE INDEX idx_papers_arxiv_id ON papers(arxiv_id);

-- Chunks table: stores text chunks with embeddings
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    section TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    embedding vector(384),  -- all-MiniLM-L6-v2 produces 384-dim vectors
    UNIQUE(paper_id, chunk_index)
);

-- Index on paper_id for fast joins
CREATE INDEX idx_chunks_paper_id ON chunks(paper_id);

-- HNSW index for fast approximate nearest neighbor search
-- Using cosine distance (vector_cosine_ops) to match the embedding model
CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Queries table: stores query history and results
CREATE TABLE queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question TEXT NOT NULL,
    answer TEXT,
    citations JSONB NOT NULL DEFAULT '[]',
    retrieved_chunks JSONB NOT NULL DEFAULT '[]',
    faithfulness_score REAL,
    faithfulness_details JSONB NOT NULL DEFAULT '{}',
    timing JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for listing recent queries
CREATE INDEX idx_queries_created_at ON queries(created_at DESC);

-- Function to get paper chunk count (used by the API)
CREATE OR REPLACE FUNCTION get_paper_with_chunk_count(paper_uuid UUID)
RETURNS TABLE (
    id UUID,
    arxiv_id TEXT,
    title TEXT,
    authors TEXT[],
    abstract TEXT,
    url TEXT,
    pdf_url TEXT,
    ingested_at TIMESTAMPTZ,
    chunk_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id,
        p.arxiv_id,
        p.title,
        p.authors,
        p.abstract,
        p.url,
        p.pdf_url,
        p.ingested_at,
        COUNT(c.id) AS chunk_count
    FROM papers p
    LEFT JOIN chunks c ON c.paper_id = p.id
    WHERE p.id = paper_uuid
    GROUP BY p.id;
END;
$$ LANGUAGE plpgsql;

-- Function to search similar chunks using cosine similarity
CREATE OR REPLACE FUNCTION search_similar_chunks(
    query_embedding vector(384),
    match_count INTEGER DEFAULT 10,
    filter_paper_ids UUID[] DEFAULT NULL
)
RETURNS TABLE (
    chunk_id UUID,
    paper_id UUID,
    paper_title TEXT,
    content TEXT,
    similarity REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id AS chunk_id,
        c.paper_id,
        p.title AS paper_title,
        c.content,
        (1 - (c.embedding <=> query_embedding))::REAL AS similarity
    FROM chunks c
    JOIN papers p ON p.id = c.paper_id
    WHERE
        c.embedding IS NOT NULL
        AND (filter_paper_ids IS NULL OR c.paper_id = ANY(filter_paper_ids))
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;
