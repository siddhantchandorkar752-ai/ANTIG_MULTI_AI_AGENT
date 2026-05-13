-- OMEGA RESEARCH GRID — PostgreSQL Init
-- Creates extension and indexes not handled by SQLAlchemy

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- Trigram search for text queries

-- Optimize JSONB columns
CREATE INDEX IF NOT EXISTS idx_audit_payload ON audit_logs USING GIN (payload);
CREATE INDEX IF NOT EXISTS idx_report_findings ON research_reports USING GIN (key_findings);

-- Text search on queries
CREATE INDEX IF NOT EXISTS idx_session_query_trgm 
ON research_sessions USING GIN (raw_query gin_trgm_ops);
