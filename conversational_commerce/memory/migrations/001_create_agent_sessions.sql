-- conversational_commerce/memory/migrations/001_create_agent_sessions.sql
-- Run once against your PostgreSQL instance before starting the application.

CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id      VARCHAR(128)    NOT NULL,
    entity_id       VARCHAR(128)    NOT NULL,
    business_unit_id VARCHAR(128)    NOT NULL,
    user_id         VARCHAR(128)    DEFAULT NULL,
    status          VARCHAR(32)     NOT NULL DEFAULT 'active',
    state           JSONB           NOT NULL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_active_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ     NOT NULL,
    total_turns     INTEGER         NOT NULL DEFAULT 0,
    metadata        JSONB           NOT NULL DEFAULT '{}',

    CONSTRAINT agent_sessions_pkey PRIMARY KEY (session_id),
    CONSTRAINT agent_sessions_status_check
        CHECK (status IN ('active', 'expired', 'completed', 'abandoned'))
);

-- Point lookup on every single request — must be instant
CREATE INDEX IF NOT EXISTS idx_agent_sessions_session_id
    ON agent_sessions (session_id);

-- TTL cleanup job scans this index — partial index keeps it small
CREATE INDEX IF NOT EXISTS idx_agent_sessions_expires_at_active
    ON agent_sessions (expires_at)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_agent_sessions_entity_status
    ON agent_sessions (entity_id, status);

-- User history reconstruction (Phase 4 personalisation)
CREATE INDEX IF NOT EXISTS idx_agent_sessions_user_id
    ON agent_sessions (user_id)
    WHERE user_id IS NOT NULL;

COMMENT ON TABLE agent_sessions IS
    'Orchestrator session state store. '
    'Owns all conversation history and agent working memory. '
    'Written/read exclusively through session_store.py — '
    'never accessed directly by agents.';