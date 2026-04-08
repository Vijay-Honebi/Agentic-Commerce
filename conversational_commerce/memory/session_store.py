# conversational_commerce/memory/session_store.py

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import asyncpg
from asyncpg import Pool, Record

from config.settings import get_settings
from observability.logger import LogEvent, get_logger
from schemas.intent import IntentType
from schemas.session import (
    AgentContext,
    ConversationTurn,
    MessageRole,
    SessionState,
    SessionStatus,
    UserContext,
)

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Connection pool — module-level singleton
# Initialised once at application startup via init_session_store().
# Every public method acquires a connection from this pool — never creates one.
# ---------------------------------------------------------------------------
_pool: Pool | None = None


async def init_session_store() -> None:
    """
    Creates the asyncpg connection pool.
    Must be called once at application startup (lifespan handler in main.py).
    Idempotent — safe to call multiple times (subsequent calls are no-ops).
    """
    global _pool

    if _pool is not None:
        return  # Already initialised

    cfg = settings.postgres

    logger.info(
        LogEvent.APP_STARTUP,
        "Initialising PostgreSQL session store connection pool",
        min_size=cfg.pool_min_size,
        max_size=cfg.pool_max_size,
    )

    _pool = await asyncpg.create_pool(
        dsn=str(cfg.dsn).replace("+asyncpg", ""),  # asyncpg uses bare postgresql://
        min_size=cfg.pool_min_size,
        max_size=cfg.pool_max_size,
        command_timeout=cfg.command_timeout_seconds,
        # Codec: asyncpg natively handles JSONB as dicts — register for safety
        init=_register_codecs,
    )

    logger.info(
        LogEvent.APP_STARTUP,
        "PostgreSQL session store ready",
        pool_min=cfg.pool_min_size,
        pool_max=cfg.pool_max_size,
    )


async def close_session_store() -> None:
    """
    Gracefully closes the connection pool.
    Called at application shutdown (lifespan handler in main.py).
    """
    global _pool

    if _pool is None:
        return

    await _pool.close()
    _pool = None

    logger.info(LogEvent.APP_SHUTDOWN, "PostgreSQL session store pool closed")


async def _register_codecs(connection: asyncpg.Connection) -> None:
    """
    Registers JSONB codec so asyncpg serialises/deserialises
    Python dicts automatically — no manual json.dumps/loads per query.
    """
    for type_name in ("jsonb", "json"):
        await connection.set_type_codec(
            type_name,
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )


def _get_pool() -> Pool:
    """
    Internal guard. Every method calls this instead of accessing _pool directly.
    Raises RuntimeError if init_session_store() was not called — never silently
    returns None and lets a downstream AttributeError confuse the stack trace.
    """
    if _pool is None:
        raise RuntimeError(
            "Session store pool is not initialised. "
            "Ensure init_session_store() is called in the application lifespan."
        )
    return _pool


@asynccontextmanager
async def _acquire() -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Context manager that acquires a pool connection and releases it on exit.
    All public methods use this — never manage connections manually.
    """
    async with _get_pool().acquire() as conn:
        yield conn


# ---------------------------------------------------------------------------
# Public API
# All methods are async. All methods are typed.
# All methods log their operation with structured events.
# ---------------------------------------------------------------------------

async def create_session(
    session_id: str,
    business_unit_id: str,
    entity_id: str,
    user_id: str | None = None,
    user_context: UserContext | None = None,
) -> SessionState:
    """
    Creates a new session and persists it to PostgreSQL.

    Called by the Orchestrator when no existing session_id is found
    in the request, or when an existing session has expired.

    Args:
        session_id: Pre-generated unique ID (generated at API boundary).
        entity_id:   Honebi entity this session belongs to.
        business_unit_id:   Honebi business unit this session belongs to.
        user_id:    Authenticated user ID. None for anonymous sessions.
        user_context: Pre-populated user preferences from profile service.
                      None → default UserContext (anonymous/guest).

    Returns:
        Freshly created SessionState, ready for Orchestrator use.
    """
    cfg = settings.session
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=cfg.ttl_seconds)

    # Build initial state
    agent_context = AgentContext(
        user=user_context or UserContext(
            user_id=user_id,
            is_authenticated=user_id is not None,
        )
    )
    state = SessionState(
        session_id=session_id,
        business_unit_id=business_unit_id,
        entity_id=entity_id,
        status=SessionStatus.ACTIVE,
        agent_context=agent_context,
        created_at=now,
        last_active_at=now,
        expires_at=expires_at,
    )

    async with _acquire() as conn:
        await conn.execute(
            """
            INSERT INTO agent_sessions (
                session_id, business_unit_id, entity_id, user_id, status,
                state, created_at, last_active_at, expires_at,
                total_turns, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            session_id,
            business_unit_id,
            entity_id,
            user_id,
            SessionStatus.ACTIVE.value,
            state.model_dump(mode="json"),   # JSONB — full state serialised
            now,
            now,
            expires_at,
            0,
            {},
        )

    logger.info(
        LogEvent.SESSION_CREATED,
        "New session created",
        session_id=session_id,
        business_unit_id=business_unit_id,
        entity_id=entity_id,
        user_id=user_id,
        expires_at=expires_at.isoformat(),
    )

    return state


async def load_session(session_id: str) -> SessionState | None:
    """
    Loads a session from PostgreSQL by session_id.

    Returns None if:
      - session_id does not exist
      - session exists but has expired (TTL exceeded)
      - session status is not ACTIVE

    The Orchestrator treats None as "start a new session".
    Expired sessions are marked as ABANDONED on load (lazy expiry).

    Args:
        session_id: Session identifier from request header/body.

    Returns:
        Hydrated SessionState or None.
    """
    async with _acquire() as conn:
        record: Record | None = await conn.fetchrow(
            """
            SELECT session_id, entity_id, business_unit_id, user_id, status,
                   state, created_at, last_active_at, expires_at, total_turns
            FROM agent_sessions
            WHERE session_id = $1
            """,
            session_id,
        )

    if record is None:
        logger.debug(
            LogEvent.SESSION_LOADED,
            "Session not found",
            session_id=session_id,
        )
        return None

    # Lazy expiry — mark abandoned if TTL exceeded
    now = datetime.now(timezone.utc)
    expires_at = record["expires_at"]

    # asyncpg returns timezone-naive datetimes for TIMESTAMPTZ in some configs
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if expires_at < now or record["status"] != SessionStatus.ACTIVE.value:
        # Fire-and-forget expiry update — don't block the response
        await _mark_session_expired(session_id)
        logger.info(
            LogEvent.SESSION_EXPIRED,
            "Session expired on load",
            session_id=session_id,
            expired_at=expires_at.isoformat(),
            status=record["status"],
        )
        return None

    # Deserialise JSONB → SessionState
    state = SessionState.model_validate(record["state"])

    logger.info(
        LogEvent.SESSION_LOADED,
        "Session loaded",
        session_id=session_id,
        total_turns=record["total_turns"],
        last_active_at=record["last_active_at"].isoformat(),
    )

    return state


async def persist_session(state: SessionState) -> None:
    """
    Persists the current SessionState back to PostgreSQL.

    Called by the Orchestrator at the END of every graph run —
    after the response is synthesized, before it is returned to the user.

    Uses UPSERT (INSERT ... ON CONFLICT DO UPDATE) so this method
    is safe for both new and existing sessions.

    Args:
        state: The fully updated SessionState after an Orchestrator run.
    """
    now = datetime.now(timezone.utc)
    cfg = settings.session
    new_expires_at = now + timedelta(seconds=cfg.ttl_seconds)

    # Extend TTL on every activity — sliding window expiry
    state.last_active_at = now
    state.expires_at = new_expires_at

    async with _acquire() as conn:
        await conn.execute(
            """
            INSERT INTO agent_sessions (
                session_id, entity_id, business_unit_id, user_id, status,
                state, created_at, last_active_at, expires_at,
                total_turns, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (session_id) DO UPDATE SET
                status          = EXCLUDED.status,
                state           = EXCLUDED.state,
                last_active_at  = EXCLUDED.last_active_at,
                expires_at      = EXCLUDED.expires_at,
                total_turns     = EXCLUDED.total_turns
            """,
            state.session_id,
            state.entity_id,
            state.business_unit_id,
            state.agent_context.user.user_id,
            state.status.value,
            state.model_dump(mode="json"),
            state.created_at,
            now,
            new_expires_at,
            state.total_turns,
            {},
        )

    logger.info(
        LogEvent.SESSION_PERSISTED,
        "Session persisted",
        session_id=state.session_id,
        total_turns=state.total_turns,
        new_expires_at=new_expires_at.isoformat(),
    )


async def append_turn(
    session_id: str,
    role: MessageRole,
    content: str,
    intent: IntentType | None = None,
    agent_used: str | None = None,
    products_shown: list[str] | None = None,
    metadata: dict | None = None,
) -> ConversationTurn:
    """
    Convenience method: loads session, appends a turn, persists.

    For simple turn-append operations where you don't need to
    modify the full state. The Orchestrator typically calls
    persist_session() directly (more efficient for full-state updates).

    Returns the constructed ConversationTurn for immediate use.
    """
    import uuid  # sortable unique ID — chronological ordering

    turn = ConversationTurn(
        turn_id=str(uuid.uuid4()),
        role=role,
        content=content,
        intent=intent,
        agent_used=agent_used,
        products_shown=products_shown or [],
        metadata=metadata or {},
    )

    state = await load_session(session_id)
    if state is None:
        raise ValueError(
            f"Cannot append turn: session '{session_id}' not found or expired."
        )

    state.add_turn(turn)
    await persist_session(state)

    return turn


async def mark_session_completed(session_id: str) -> None:
    """
    Marks a session as COMPLETED after a successful purchase.
    Phase 3: completed sessions are the positive training signal
    for promotion and recommendation models.
    """
    await _update_session_status(session_id, SessionStatus.COMPLETED)
    logger.info(
        LogEvent.SESSION_PERSISTED,
        "Session marked as completed",
        session_id=session_id,
    )


async def mark_session_abandoned(session_id: str) -> None:
    """
    Marks a session as ABANDONED.
    Phase 3: abandoned sessions with non-empty carts trigger
    win-back promotion campaigns.
    """
    await _update_session_status(session_id, SessionStatus.ABANDONED)
    logger.info(
        LogEvent.SESSION_EXPIRED,
        "Session marked as abandoned",
        session_id=session_id,
    )


async def cleanup_expired_sessions(batch_size: int = 500) -> int:
    """
    Bulk-marks expired active sessions as ABANDONED.
    Designed to be called by a scheduled job (e.g. APScheduler / cron).
    Processes in batches to avoid long-running transactions.

    Returns:
        Number of sessions cleaned up in this run.
    """
    now = datetime.now(timezone.utc)
    total_cleaned = 0

    async with _acquire() as conn:
        while True:
            result = await conn.execute(
                """
                UPDATE agent_sessions
                SET status = $1
                WHERE session_id IN (
                    SELECT session_id
                    FROM agent_sessions
                    WHERE status = $2
                      AND expires_at < $3
                    LIMIT $4
                )
                """,
                SessionStatus.ABANDONED.value,
                SessionStatus.ACTIVE.value,
                now,
                batch_size,
            )

            # asyncpg returns "UPDATE N" as string
            affected = int(result.split()[-1])
            total_cleaned += affected

            if affected < batch_size:
                break  # No more rows to process

    logger.info(
        LogEvent.SESSION_EXPIRED,
        "Expired session cleanup complete",
        sessions_cleaned=total_cleaned,
    )

    return total_cleaned


# ---------------------------------------------------------------------------
# Internal helpers — not part of public API
# ---------------------------------------------------------------------------

async def _mark_session_expired(session_id: str) -> None:
    """Lazy expiry: called when an expired session is accessed."""
    await _update_session_status(session_id, SessionStatus.ABANDONED)


async def _update_session_status(
    session_id: str,
    status: SessionStatus,
) -> None:
    async with _acquire() as conn:
        await conn.execute(
            """
            UPDATE agent_sessions
            SET status = $1, last_active_at = $2
            WHERE session_id = $3
            """,
            status.value,
            datetime.now(timezone.utc),
            session_id,
        )