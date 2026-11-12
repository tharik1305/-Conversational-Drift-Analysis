"""
Plotly visualization charts for conversational drift metrics.
"""

from typing import List
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from conversational_drift.models import ConversationReport, DriftStatus


def create_drift_trajectory_chart(report: ConversationReport) -> go.Figure:
    """
    Creates an interactive Plotly trajectory chart showing anchor distance,
    drift threshold zones, inflection points, and turn-to-turn local distance.
    """
    turns = report.turns
    turn_indices = [t.turn_index for t in turns]
    anchor_dists = [t.anchor_distance for t in turns]
    step_dists = [t.stepwise_distance for t in turns]
    speakers = [t.speaker for t in turns]
    texts = [t.text for t in turns]
    statuses = [t.drift_status.value for t in turns]

    # Custom hover text
    hover_texts = [
        f"<b>Turn {t.turn_index}</b> ({t.speaker})<br>"
        f"<b>Status:</b> {t.drift_status.value}<br>"
        f"<b>Anchor Dist:</b> {t.anchor_distance:.3f}<br>"
        f"<b>Step Dist:</b> {t.stepwise_distance:.3f}<br>"
        f"<b>Keywords:</b> {', '.join(t.keywords[:3])}<br>"
        f"<i>\"{t.text[:80]}...\"</i>"
        for t in turns
    ]

    fig = go.Figure()

    # Threshold horizontal reference bands
    max_x = max(turn_indices) if turn_indices else 10
    fig.add_hrect(y0=0.0, y1=0.35, fillcolor="rgba(16, 185, 129, 0.08)", line_width=0, layer="below", annotation_text="ON TOPIC (Anchor)", annotation_position="top left")
    fig.add_hrect(y0=0.35, y1=0.60, fillcolor="rgba(59, 130, 246, 0.08)", line_width=0, layer="below", annotation_text="MILD ELABORATION", annotation_position="top left")
    fig.add_hrect(y0=0.60, y1=0.80, fillcolor="rgba(245, 158, 11, 0.10)", line_width=0, layer="below", annotation_text="MODERATE DRIFT", annotation_position="top left")
    fig.add_hrect(y0=0.80, y1=1.40, fillcolor="rgba(239, 68, 68, 0.12)", line_width=0, layer="below", annotation_text="SEVERE DIVERGENCE", annotation_position="top left")

    # Step-wise distance trace (dashed line)
    fig.add_trace(go.Scatter(
        x=turn_indices,
        y=step_dists,
        mode="lines+markers",
        name="Step-wise Local Distance",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        marker=dict(size=5, color="#94a3b8"),
        opacity=0.75,
        hoverinfo="skip"
    ))

    # Main Anchor Distance Trajectory
    fig.add_trace(go.Scatter(
        x=turn_indices,
        y=anchor_dists,
        mode="lines+markers",
        name="Anchor Drift Distance",
        line=dict(color="#6366f1", width=3.5, shape="spline"),
        marker=dict(
            size=9,
            color=anchor_dists,
            colorscale=[[0, "#10b981"], [0.4, "#3b82f6"], [0.7, "#f59e0b"], [1, "#ef4444"]],
            showscale=False
        ),
        text=hover_texts,
        hoverinfo="text"
    ))

    # Inflection points markers (anomalies)
    inflection_indices = [t.turn_index for t in turns if t.is_inflection_point]
    inflection_dists = [t.anchor_distance for t in turns if t.is_inflection_point]
    inflection_hover = [
        f"<b>⚠️ INFLECTION ALERT (Turn {t.turn_index})</b><br>"
        f"Speaker: {t.speaker}<br>"
        f"Velocity: +{t.drift_velocity:.3f}<br>"
        f"Utterance: \"{t.text[:70]}...\""
        for t in turns if t.is_inflection_point
    ]

    if inflection_indices:
        fig.add_trace(go.Scatter(
            x=inflection_indices,
            y=inflection_dists,
            mode="markers",
            name="Inflection Breakaway Point",
            marker=dict(
                symbol="star",
                size=16,
                color="#dc2626",
                line=dict(width=2, color="#ffffff")
            ),
            text=inflection_hover,
            hoverinfo="text"
        ))

    fig.update_layout(
        title="<b>Conversational Drift Trajectory</b> (Semantic Distance over Turns)",
        xaxis_title="Turn Index (#)",
        yaxis_title="Semantic Distance from Anchor (1 - Cosine Sim)",
        template="plotly_dark",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        yaxis=dict(range=[0.0, max(1.15, max(anchor_dists) * 1.1 if anchor_dists else 1.0)]),
        height=450
    )

    return fig


def create_kinematics_chart(report: ConversationReport) -> go.Figure:
    """
    Subplots showing Velocity (rate of drift) and Acceleration.
    """
    turns = report.turns
    indices = [t.turn_index for t in turns]
    vels = [t.drift_velocity for t in turns]
    accs = [t.drift_acceleration for t in turns]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Drift Velocity (Δ Distance / Turn)", "Drift Acceleration (Δ² Distance / Turn²)"),
        vertical_spacing=0.12
    )

    # Velocity Bar Chart
    vel_colors = ["#ef4444" if v > 0.15 else ("#3b82f6" if v >= 0 else "#10b981") for v in vels]
    fig.add_trace(
        go.Bar(x=indices, y=vels, name="Velocity", marker_color=vel_colors),
        row=1, col=1
    )

    # Acceleration Line Chart
    fig.add_trace(
        go.Scatter(x=indices, y=accs, name="Acceleration", line=dict(color="#f59e0b", width=2), mode="lines+markers"),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis2_title="Turn Index (#)"
    )

    return fig


def create_2d_trajectory_chart(report: ConversationReport, coords_2d: np.ndarray) -> go.Figure:
    """
    2D PCA semantic trajectory chart mapping how the conversation traveled across semantic space.
    """
    turns = report.turns
    n = len(turns)

    if coords_2d.shape[0] < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough data points for 2D semantic projection.")
        return fig

    x = coords_2d[:, 0]
    y = coords_2d[:, 1]

    hover_texts = [
        f"<b>Turn {t.turn_index}</b> ({t.speaker})<br>"
        f"<b>Cluster:</b> {t.cluster_id}<br>"
        f"<b>Keywords:</b> {', '.join(t.keywords[:3])}<br>"
        f"<i>\"{t.text[:60]}...\"</i>"
        for t in turns
    ]

    fig = go.Figure()

    # Direction path line connecting turns
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color="rgba(148, 163, 184, 0.45)", width=2, dash="dash"),
        name="Path",
        hoverinfo="skip"
    ))

    # Scatter points colored by turn index
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        name="Turns",
        marker=dict(
            size=14,
            color=list(range(n)),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Turn #")
        ),
        text=[f"#{i}" for i in range(n)],
        textposition="top center",
        hovertext=hover_texts,
        hoverinfo="text"
    ))

    # Highlight Start (Anchor) and Finish
    fig.add_trace(go.Scatter(
        x=[x[0]], y=[y[0]],
        mode="markers",
        marker=dict(size=20, color="#10b981", symbol="circle-open", line=dict(width=3, color="#10b981")),
        name="Anchor (Start)",
        hovertext=f"START: {turns[0].text[:50]}...",
        hoverinfo="text"
    ))

    fig.add_trace(go.Scatter(
        x=[x[-1]], y=[y[-1]],
        mode="markers",
        marker=dict(size=20, color="#ef4444", symbol="square-open", line=dict(width=3, color="#ef4444")),
        name="Termination (End)",
        hovertext=f"END: {turns[-1].text[:50]}...",
        hoverinfo="text"
    ))

    fig.update_layout(
        title="<b>2D Semantic Space Trajectory (PCA Projection)</b>",
        xaxis_title="Semantic Dimension 1 (PCA)",
        yaxis_title="Semantic Dimension 2 (PCA)",
        template="plotly_dark",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig
