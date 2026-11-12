"""
Conversational Drift Analyzer
=============================
An end-to-end NLP framework to measure, detect, and visualize conversational drift in multi-turn dialogues.
"""

from .models import (
    Turn,
    TurnMetrics,
    Conversation,
    ConversationReport,
    DriftStatus,
    DriftZone,
)
from .analyzer import DriftAnalyzer
from .reporter import DriftReporter
from .utils import (
    parse_raw_transcript,
    load_conversation_from_json,
    report_to_dataframe,
)

__version__ = "1.0.0"

__all__ = [
    "DriftAnalyzer",
    "Conversation",
    "Turn",
    "TurnMetrics",
    "ConversationReport",
    "DriftStatus",
    "DriftZone",
    "DriftReporter",
    "parse_raw_transcript",
    "load_conversation_from_json",
    "report_to_dataframe",
]
