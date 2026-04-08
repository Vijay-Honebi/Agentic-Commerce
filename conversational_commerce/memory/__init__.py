# conversational_commerce/memory/__init__.py

from memory.session_store import (
    append_turn,
    cleanup_expired_sessions,
    close_session_store,
    create_session,
    init_session_store,
    load_session,
    mark_session_abandoned,
    mark_session_completed,
    persist_session,
)

__all__ = [
    "init_session_store",
    "close_session_store",
    "create_session",
    "load_session",
    "persist_session",
    "append_turn",
    "mark_session_completed",
    "mark_session_abandoned",
    "cleanup_expired_sessions",
]

# A persistent, scalable, AI session memory system that supports:

# multi-turn conversations
# multi-agent orchestration
# future personalization
# analytics
# TTL + lifecycle management

# This is NOT chat history — this is stateful AI memory infrastructure