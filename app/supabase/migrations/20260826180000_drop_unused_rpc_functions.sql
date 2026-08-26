-- Drop RPC functions that no code path calls.
--
-- The initial migration defined get_paper_with_chunk_count() and
-- search_similar_chunks() as "used by the API", but PostgresVectorStore
-- inlines its own SQL for both operations and nothing in app/src references
-- either function. On Supabase they stayed exposed as PostgREST RPC endpoints,
-- so they were reachable surface area guaranteed to drift from the queries the
-- application actually runs — search_similar_chunks in particular hardcodes a
-- 384-dimension vector and its own filter semantics.

DROP FUNCTION IF EXISTS get_paper_with_chunk_count(UUID);
DROP FUNCTION IF EXISTS search_similar_chunks(vector, INTEGER, UUID[]);
