# conversational_commerce/schemas/session.py

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from schemas.intent import IntentType
from schemas.product import ProductCard


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationTurn(BaseModel):
    """
    A single exchange in the conversation history.

    Stored as part of SessionState. The last N turns (configured by
    SESSION_MAX_TURNS_IN_MEMORY) are included in every LLM context window.
    Older turns are truncated from the LLM context but remain in the DB
    for analytics and personalization (Phase 4).

    Design note:
        We store structured metadata alongside the text so Phase 3/4
        can mine conversation history for intent patterns without
        replaying every message through the LLM.
    """

    turn_id: str
    role: MessageRole
    content: str

    # Structured metadata — populated by orchestrator, not the LLM
    intent: IntentType | None = Field(
        default=None,
        description="Classified intent for user turns. None for assistant turns.",
    )
    agent_used: str | None = Field(
        default=None,
        description="Which specialist agent handled this turn. "
                    "e.g. 'discovery_agent'. None for clarification turns.",
    )
    products_shown: list[str] = Field(
        default_factory=list,
        description="product_ids shown to user in this turn. "
                    "Used by Phase 3 to avoid re-showing same products.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-turn observability data: token counts, latency, etc.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class DiscoveryContext(BaseModel):
    """
    Discovery-specific context within a session.

    The Orchestrator maintains this across turns so it can:
      - Avoid showing the same products twice
      - Understand what the user has already seen and rejected
      - Build progressive refinement across multiple queries
        e.g. Turn 1: "show me shoes" → Turn 2: "now show me red ones"
             The second query inherits category=shoes as a constraint.
    """

    last_query: str = Field(
        default="",
        description="Last semantic query string. Used for refinement turns.",
    )
    last_filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Last applied HardConstraints as dict. "
                    "Inherited by refinement queries.",
    )
    shown_product_ids: list[str] = Field(
        default_factory=list,
        description="All product_ids shown in this session. "
                    "Exclusion list for subsequent searches.",
    )
    last_results: list[ProductCard] = Field(
        default_factory=list,
        description="Most recent search results. "
                    "Orchestrator references these when user says "
                    "'tell me more about the first one'.",
    )
    total_searches: int = Field(
        default=0,
        description="Number of search turns in this session. "
                    "Phase 3 uses this to decide when to push a promotion.",
    )


class CartContext(BaseModel):
    """
    Phase 2 placeholder — scaffolded now so SessionState schema doesn't
    change when Phase 2 is implemented.
    Cart Agent writes here. Checkout Agent reads from here.
    Orchestrator reads this to inform Phase 3 promotion decisions.
    """

    cart_id: str | None = None
    item_count: int = Field(default=0)
    cart_total: float = Field(default=0.0)
    currency: str = Field(default="INR")
    last_updated: datetime | None = None
    # Full cart items stored here in Phase 2
    items: list[dict[str, Any]] = Field(default_factory=list)


class UserContext(BaseModel):
    """
    Persistent user attributes known before the session starts.
    Populated from user profile at session creation.
    Phase 4: this grows significantly with behavioral data.
    """

    user_id: str | None = None
    is_authenticated: bool = Field(default=False)
    preferred_language: str = Field(default="en")
    preferred_currency: str = Field(default="INR")

    # Phase 3/4: preference signals
    preferred_brands: list[str] = Field(default_factory=list)
    preferred_categories: list[str] = Field(default_factory=list)
    price_sensitivity: str | None = Field(
        default=None,
        description="Inferred from purchase history. "
                    "One of: 'budget', 'mid_range', 'premium'. "
                    "Used by Phase 4 personalization.",
    )


class AgentContext(BaseModel):
    """
    Full context block passed to context_builder.py when constructing
    an AgentRequest. The builder selects the relevant slice for each agent.

    This is the Orchestrator's complete working memory about the current session.
    """

    discovery: DiscoveryContext = Field(default_factory=DiscoveryContext)
    cart: CartContext = Field(default_factory=CartContext)
    user: UserContext = Field(default_factory=UserContext)


class SessionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    COMPLETED = "completed"      # User completed a purchase
    ABANDONED = "abandoned"      # Session expired without purchase (Phase 3 signal)


class SessionState(BaseModel):
    """
    The Orchestrator's single source of truth for a conversation.

    Owned exclusively by the Orchestrator.
    Written and read only through session_store.py.
    Agents never access SessionState directly — they receive
    curated slices via AgentRequest.context.

    Persistence:
        Serialised as JSON and stored in PostgreSQL (agent_sessions table).
        Loaded at the start of every Orchestrator run.
        Persisted at the end of every Orchestrator run.
        TTL enforced by SESSION_TTL_SECONDS setting.

    Phase evolution:
        Phase 1 → discovery context populated
        Phase 2 → cart context populated
        Phase 3 → promotion_history added
        Phase 4 → behavioral_signals added (append-only log)
    """

    session_id: str = Field(description="Globally unique session identifier.")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE)

    # Conversation history (full — never truncated in DB)
    turns: list[ConversationTurn] = Field(
        default_factory=list,
        description="Complete conversation history for this session.",
    )

    # Structured working memory
    agent_context: AgentContext = Field(default_factory=AgentContext)

    # Session lifecycle
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    last_active_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    expires_at: datetime | None = None

    # Analytics
    business_unit_id: str | None = None
    entity_id: str | None = None
    total_turns: int = Field(default=0)

    def get_recent_turns(self, max_turns: int) -> list[ConversationTurn]:
        """
        Returns the last `max_turns` turns for LLM context window.
        Older turns are preserved in DB but excluded from LLM context
        to avoid token overflow on long sessions.
        """
        return self.turns[-max_turns:] if len(self.turns) > max_turns else self.turns

    def add_turn(self, turn: ConversationTurn) -> None:
        """Appends a turn and updates session activity tracking."""
        self.turns.append(turn)
        self.total_turns += 1
        self.last_active_at = datetime.now(timezone.utc)

    def get_context_for_llm(self, max_turns: int) -> list[dict[str, str]]:
        """
        Formats recent turns as OpenAI message dicts for LLM context.
        Only includes role + content — strips internal metadata.
        """
        return [
            {"role": turn.role.value, "content": turn.content}
            for turn in self.get_recent_turns(max_turns)
        ]