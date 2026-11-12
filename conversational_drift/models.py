"""
Data models and schemas for the Conversational Drift Analyzer.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class DriftStatus(str, Enum):
    ON_TOPIC = "ON_TOPIC"
    MILD_ELABORATION = "MILD_ELABORATION"
    MODERATE_DRIFT = "MODERATE_DRIFT"
    SEVERE_DRIFT = "SEVERE_DRIFT"


class Turn(BaseModel):
    """Represents a single conversational turn (utterance)."""
    turn_index: int = Field(..., description="Zero-based or 1-based sequential index of the turn")
    speaker: str = Field(..., description="Speaker identifier (e.g. 'User', 'Assistant', 'Agent', 'Customer')")
    text: str = Field(..., description="The spoken or typed text content of the turn")
    timestamp: Optional[str] = Field(None, description="Optional ISO timestamp or relative time string")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata for the turn")


class TurnMetrics(BaseModel):
    """NLP metrics calculated for a single conversation turn."""
    turn_index: int
    speaker: str
    text: str
    
    # Distance & Similarity to Anchor (turn 0 or custom goal)
    anchor_similarity: float = Field(..., description="Cosine similarity [0..1] with conversation anchor/goal")
    anchor_distance: float = Field(..., description="Semantic distance [0..1] from anchor (1 - similarity)")
    
    # Step-wise Local Coherence
    stepwise_similarity: float = Field(..., description="Cosine similarity [0..1] with immediately preceding turn")
    stepwise_distance: float = Field(..., description="Semantic distance [0..1] from preceding turn")
    
    # Context Window Coherence (Rolling average of prior k turns)
    context_similarity: float = Field(..., description="Cosine similarity [0..1] with recent sliding context window")
    
    # Drift Kinematics
    drift_velocity: float = Field(..., description="Rate of change in anchor distance per turn")
    drift_acceleration: float = Field(..., description="Acceleration of drift per turn")
    
    # Categorization & Anomaly Detection
    drift_status: DriftStatus = Field(..., description="Discrete classification of drift severity")
    is_inflection_point: bool = Field(False, description="True if this turn marks an abrupt divergence point")
    
    # Semantic & Topic Insights
    keywords: List[str] = Field(default_factory=list, description="Extracted salient keywords for this turn")
    cluster_id: int = Field(0, description="Semantic cluster ID assigned to this turn")
    cluster_label: Optional[str] = Field(None, description="Descriptive label or top terms for the cluster")


class Conversation(BaseModel):
    """Collection of turns representing an entire dialogue session."""
    conversation_id: str = Field(..., description="Unique identifier for the conversation session")
    turns: List[Turn] = Field(default_factory=list, description="Ordered list of dialogue turns")
    anchor_goal: Optional[str] = Field(None, description="Explicit objective or goal. If None, turn 0 is used as anchor")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session-level metadata")

    def add_turn(self, speaker: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Turn:
        idx = len(self.turns)
        turn = Turn(turn_index=idx, speaker=speaker, text=text, metadata=metadata or {})
        self.turns.append(turn)
        return turn


class DriftZone(BaseModel):
    """A contiguous sequence of turns exhibiting persistent drift."""
    start_turn: int
    end_turn: int
    severity: DriftStatus
    dominant_keywords: List[str]
    description: str


class ConversationReport(BaseModel):
    """Comprehensive analysis summary for an entire dialogue."""
    conversation_id: str
    total_turns: int
    anchor_text: str
    
    # Global Health & Cohesion Scores
    cohesion_index: float = Field(..., description="Overall conversational cohesion score (0 to 100%)")
    mean_anchor_distance: float = Field(..., description="Average semantic distance from the anchor")
    max_drift_turn: int = Field(..., description="Turn index where maximum drift occurred")
    max_drift_distance: float = Field(..., description="Peak anchor distance observed")
    
    # Divergence Insights
    first_inflection_turn: Optional[int] = Field(None, description="First turn where significant drift was initiated")
    drift_zones: List[DriftZone] = Field(default_factory=list, description="Identified contiguous drift segments")
    
    # Turn-by-Turn Metrics
    turns: List[TurnMetrics] = Field(default_factory=list)
    
    # Global Clusters
    cluster_summaries: Dict[int, List[str]] = Field(default_factory=dict)
    
    # Natural Language Summary
    executive_summary: str = Field(..., description="Human-readable synthesis of conversation trajectory and drift")
