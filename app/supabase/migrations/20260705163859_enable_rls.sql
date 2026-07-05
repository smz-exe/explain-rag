-- Enable Row-Level Security (RLS) on all public tables.
--
-- Why: Supabase auto-exposes every table in the `public` schema through the
-- PostgREST API (https://<project-ref>.supabase.co/rest/v1/...). With RLS
-- disabled, any holder of the anon key can read/edit/delete these tables.
-- This closes that hole (Supabase advisor: rls_disabled_in_public).
--
-- Safety: No policies are created, so anon/authenticated roles are fully
-- denied via PostgREST. The application connects as the `postgres` table-owner
-- role (DATABASE_URL pooler connection), which bypasses RLS unless FORCE is
-- set — so the backend is unaffected.

ALTER TABLE public.papers  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chunks  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.queries ENABLE ROW LEVEL SECURITY;
