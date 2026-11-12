"""
Report generator for Conversational Drift Analysis (Markdown, HTML, JSON).
"""

import json
from pathlib import Path
from typing import Optional
from .models import ConversationReport, DriftStatus


class DriftReporter:
    """Formats and exports conversation drift reports to various media formats."""

    @staticmethod
    def to_markdown(report: ConversationReport) -> str:
        """Generates a clean GitHub-flavored Markdown report."""
        lines = [
            f"# Conversational Drift Report: `{report.conversation_id}`",
            "",
            "## Executive Summary",
            f"> {report.executive_summary}",
            "",
            "## Global Metrics",
            f"- **Overall Cohesion Index**: `{report.cohesion_index:.1f}%`",
            f"- **Total Turns**: `{report.total_turns}`",
            f"- **Anchor Statement**: *\"{report.anchor_text}\"*",
            f"- **Mean Anchor Distance**: `{report.mean_anchor_distance:.3f}`",
            f"- **Peak Drift Turn**: `Turn {report.max_drift_turn}` (Distance: `{report.max_drift_distance:.3f}`)",
            f"- **First Inflection Turn**: `{report.first_inflection_turn if report.first_inflection_turn is not None else 'None'}`",
            "",
            "## Identified Drift Zones",
        ]

        if not report.drift_zones:
            lines.append("*No persistent drift zones detected.*")
        else:
            for z in report.drift_zones:
                kws = ", ".join(z.dominant_keywords) if z.dominant_keywords else "N/A"
                lines.append(f"- **Turns {z.start_turn} – {z.end_turn}** [{z.severity.value}] | *Topics: {kws}*")

        lines.extend([
            "",
            "## Turn-by-Turn Trajectory",
            "| # | Speaker | Text Snippet | Anchor Dist | Step Dist | Drift Status | Inflection |",
            "|---|---|---|---|---|---|---|",
        ])

        for t in report.turns:
            snippet = (t.text[:45] + "...") if len(t.text) > 45 else t.text
            clean_snippet = snippet.replace("|", "\\|").replace("\n", " ")
            flag = "⚠️ YES" if t.is_inflection_point else "-"
            lines.append(
                f"| {t.turn_index} | **{t.speaker}** | {clean_snippet} | {t.anchor_distance:.3f} | {t.stepwise_distance:.3f} | `{t.drift_status.value}` | {flag} |"
            )

        return "\n".join(lines)

    @staticmethod
    def to_json(report: ConversationReport, indent: int = 2) -> str:
        """Serializes the report into a JSON formatted string."""
        return report.model_dump_json(indent=indent)

    @staticmethod
    def to_html(report: ConversationReport) -> str:
        """Generates a standalone, beautiful HTML dashboard report."""
        # Color mapper
        color_map = {
            DriftStatus.ON_TOPIC.value: "#10b981",
            DriftStatus.MILD_ELABORATION.value: "#3b82f6",
            DriftStatus.MODERATE_DRIFT.value: "#f59e0b",
            DriftStatus.SEVERE_DRIFT.value: "#ef4444",
        }

        table_rows = []
        for t in report.turns:
            badge_color = color_map.get(t.drift_status.value, "#6b7280")
            inflection_badge = '<span class="badge danger">ALERT</span>' if t.is_inflection_point else ""
            kws = ", ".join(t.keywords)
            table_rows.append(f"""
            <tr>
                <td><strong>#{t.turn_index}</strong></td>
                <td><span class="speaker-pill">{t.speaker}</span></td>
                <td class="text-cell">{t.text}</td>
                <td><strong>{t.anchor_distance:.3f}</strong></td>
                <td>{t.stepwise_distance:.3f}</td>
                <td><span class="status-badge" style="background-color: {badge_color}20; color: {badge_color}; border: 1px solid {badge_color};">{t.drift_status.value}</span></td>
                <td>{inflection_badge}</td>
                <td class="keyword-cell">{kws}</td>
            </tr>
            """)

        rows_html = "\n".join(table_rows)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conversational Drift Report - {report.conversation_id}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #6366f1;
            --accent: #38bdf8;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 32px 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 24px;
            margin-bottom: 28px;
        }}
        h1 {{
            font-size: 28px;
            margin: 0 0 8px 0;
            color: #ffffff;
            font-weight: 700;
        }}
        .meta-subtitle {{
            color: var(--text-muted);
            font-size: 14px;
        }}
        .grid-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }}
        .card-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        .card-value {{
            font-size: 26px;
            font-weight: 700;
            color: var(--accent);
        }}
        .summary-box {{
            background: #1e293b80;
            border-left: 4px solid var(--primary);
            padding: 18px 24px;
            border-radius: 6px;
            margin-bottom: 32px;
            line-height: 1.6;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--surface);
            border-radius: 10px;
            overflow: hidden;
            font-size: 14px;
        }}
        th, td {{
            padding: 14px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background-color: #111827;
            color: var(--text-muted);
            font-weight: 600;
            font-size: 13px;
        }}
        tr:hover {{
            background-color: #243248;
        }}
        .text-cell {{
            max-width: 380px;
            color: #e2e8f0;
            line-height: 1.4;
        }}
        .speaker-pill {{
            background: #334155;
            padding: 3px 8px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .status-badge {{
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .badge.danger {{
            background: #dc2626;
            color: #fff;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
        }}
        .keyword-cell {{
            color: #38bdf8;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Conversational Drift Analysis</h1>
            <div class="meta-subtitle">Session ID: <code>{report.conversation_id}</code> | Anchor: "{report.anchor_text}"</div>
        </div>

        <div class="grid-cards">
            <div class="card">
                <div class="card-label">Cohesion Index</div>
                <div class="card-value" style="color: {'#10b981' if report.cohesion_index >= 70 else '#f59e0b'};">{report.cohesion_index:.1f}%</div>
            </div>
            <div class="card">
                <div class="card-label">Total Turns</div>
                <div class="card-value">{report.total_turns}</div>
            </div>
            <div class="card">
                <div class="card-label">Mean Semantic Distance</div>
                <div class="card-value">{report.mean_anchor_distance:.3f}</div>
            </div>
            <div class="card">
                <div class="card-label">Peak Drift Turn</div>
                <div class="card-value">Turn #{report.max_drift_turn}</div>
            </div>
        </div>

        <div class="summary-box">
            <strong>Executive Insights:</strong><br>
            {report.executive_summary}
        </div>

        <h2>Turn Trajectory Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Turn</th>
                    <th>Speaker</th>
                    <th>Utterance</th>
                    <th>Anchor Dist</th>
                    <th>Step Dist</th>
                    <th>Status</th>
                    <th>Alert</th>
                    <th>Extracted Keyphrases</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        return html

    @classmethod
    def save_report(cls, report: ConversationReport, output_path: str, format_type: str = "html") -> Path:
        """Saves report to disk in the specified format ('html', 'md', 'json')."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if format_type.lower() == "html":
            content = cls.to_html(report)
        elif format_type.lower() in ("md", "markdown"):
            content = cls.to_markdown(report)
        elif format_type.lower() == "json":
            content = cls.to_json(report)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        with open(out_p, "w", encoding="utf-8") as f:
            f.write(content)

        return out_p
