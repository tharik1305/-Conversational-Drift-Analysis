"""
Table and badge formatting components for Streamlit UI.
"""

from typing import List
import pandas as pd
from conversational_drift.models import TurnMetrics, DriftStatus


def render_drift_badge_html(status: DriftStatus) -> str:
    """Returns HTML for color-coded status badge."""
    colors = {
        DriftStatus.ON_TOPIC: ("#10b981", "rgba(16, 185, 129, 0.15)"),
        DriftStatus.MILD_ELABORATION: ("#3b82f6", "rgba(59, 130, 246, 0.15)"),
        DriftStatus.MODERATE_DRIFT: ("#f59e0b", "rgba(245, 158, 11, 0.15)"),
        DriftStatus.SEVERE_DRIFT: ("#ef4444", "rgba(239, 68, 68, 0.15)"),
    }
    fg, bg = colors.get(status, ("#94a3b8", "rgba(148, 163, 184, 0.15)"))
    return f'<span style="background-color: {bg}; color: {fg}; border: 1px solid {fg}; padding: 3px 8px; border-radius: 9999px; font-size: 11px; font-weight: 700;">{status.value}</span>'


def build_dataframe_for_ui(turns: List[TurnMetrics]) -> pd.DataFrame:
    """Creates a cleaned DataFrame ready for Streamlit's st.dataframe with custom styling."""
    rows = []
    for t in turns:
        rows.append({
            "Turn": f"#{t.turn_index}",
            "Speaker": t.speaker,
            "Text": t.text,
            "Anchor Dist": f"{t.anchor_distance:.3f}",
            "Step Dist": f"{t.stepwise_distance:.3f}",
            "Status": t.drift_status.value,
            "Inflection Alert": "🚨 ALERT" if t.is_inflection_point else "—",
            "Key Terms": ", ".join(t.keywords),
        })
    return pd.DataFrame(rows)
