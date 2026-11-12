"""
End-to-end tests for DriftAnalyzer.
"""

import pytest
from conversational_drift import DriftAnalyzer, Conversation, Turn
from conversational_drift.models import DriftStatus
from conversational_drift.reporter import DriftReporter


def test_analyzer_end_to_end_drifting():
    analyzer = DriftAnalyzer(embedder="tfidf")

    conv = Conversation(
        conversation_id="test_drift_session",
        anchor_goal="Solve Python syntax error in dictionary comprehension"
    )
    conv.add_turn("User", "How do I fix a SyntaxError in my Python dictionary comprehension?")
    conv.add_turn("Assistant", "Make sure you have both key and value separated by colon inside the expression.")
    conv.add_turn("User", "What is the capital city of Australia?")
    conv.add_turn("Assistant", "The capital city of Australia is Canberra.")
    conv.add_turn("User", "Do kangaroos live in Canberra parks?")

    report = analyzer.analyze(conv)

    assert report.total_turns == 5
    assert report.cohesion_index < 90.0
    assert report.max_drift_turn >= 2
    # Check that turn 2 or 3 has moderate or severe drift
    assert any(t.drift_status in (DriftStatus.MODERATE_DRIFT, DriftStatus.SEVERE_DRIFT) for t in report.turns)

    # Check reporter export
    html_output = DriftReporter.to_html(report)
    assert "<!DOCTYPE html>" in html_output
    assert report.conversation_id in html_output

    md_output = DriftReporter.to_markdown(report)
    assert "# Conversational Drift Report" in md_output


def test_analyzer_anchored_conversation():
    analyzer = DriftAnalyzer(embedder="tfidf")

    conv = Conversation(
        conversation_id="test_anchored_session",
        anchor_goal="Database connection pooling in PostgreSQL"
    )
    conv.add_turn("Dev", "How should I configure connection pooling for PostgreSQL?")
    conv.add_turn("DBA", "You can use PgBouncer or configure max_connections in postgresql.conf.")
    conv.add_turn("Dev", "What is the recommended pool mode for PgBouncer?")
    conv.add_turn("DBA", "Transaction pooling mode is typically recommended for web services.")

    report = analyzer.analyze(conv)

    # Anchored conversation should have high cohesion index
    assert report.total_turns == 4
    assert report.cohesion_index >= 50.0
