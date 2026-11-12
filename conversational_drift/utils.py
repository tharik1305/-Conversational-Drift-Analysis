"""
Utility functions for parsing, formatting, and serializing conversation datasets.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
from .models import Conversation, Turn, ConversationReport


def parse_raw_transcript(
    text: str,
    conversation_id: str = "transcript_001",
    anchor_goal: Optional[str] = None
) -> Conversation:
    """
    Parses unformatted or formatted raw conversation transcripts into a structured Conversation.
    Supports formats like:
      Speaker: Hello there
      User: Can you help?
      [Timestamp] Agent: Sure!
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    conv = Conversation(conversation_id=conversation_id, anchor_goal=anchor_goal)

    # Regex pattern matching 'Speaker: utterance' or '[time] Speaker: utterance'
    speaker_pattern = re.compile(r"^(?:\[[\w\:\s\.\-]+\]\s*)?([A-Za-z0-9_\s]{1,30})\s*:\s*(.*)$")

    current_speaker = "Participant"
    turn_idx = 0

    for line in lines:
        match = speaker_pattern.match(line)
        if match:
            speaker = match.group(1).strip()
            content = match.group(2).strip()
            conv.turns.append(Turn(turn_index=turn_idx, speaker=speaker, text=content))
            turn_idx += 1
            current_speaker = speaker
        else:
            # Multi-line utterance continuation or unlabelled turn
            if conv.turns:
                conv.turns[-1].text += f" {line}"
            else:
                conv.turns.append(Turn(turn_index=turn_idx, speaker=current_speaker, text=line))
                turn_idx += 1

    return conv


def load_conversation_from_json(file_path: str) -> Conversation:
    """Loads a conversation from a JSON file."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        conv_id = data.get("conversation_id", path.stem)
        anchor_goal = data.get("anchor_goal", None)
        turns_data = data.get("turns", [])
        
        conv = Conversation(conversation_id=conv_id, anchor_goal=anchor_goal)
        for i, t in enumerate(turns_data):
            if isinstance(t, dict):
                speaker = t.get("speaker", f"Speaker_{i}")
                text = t.get("text", "")
                meta = t.get("metadata", {})
                conv.turns.append(Turn(turn_index=i, speaker=speaker, text=text, metadata=meta))
            elif isinstance(t, str):
                conv.turns.append(Turn(turn_index=i, speaker="Speaker", text=t))
        return conv
    elif isinstance(data, list):
        conv = Conversation(conversation_id=path.stem)
        for i, t in enumerate(data):
            if isinstance(t, dict):
                conv.turns.append(Turn(
                    turn_index=i,
                    speaker=t.get("speaker", f"Speaker_{i}"),
                    text=t.get("text", "")
                ))
            elif isinstance(t, str):
                conv.turns.append(Turn(turn_index=i, speaker="Speaker", text=t))
        return conv
    else:
        raise ValueError("Invalid JSON format for conversation.")


def report_to_dataframe(report: ConversationReport) -> pd.DataFrame:
    """Converts the turn-by-turn metrics of a ConversationReport into a pandas DataFrame."""
    records = []
    for t in report.turns:
        records.append({
            "Turn": t.turn_index,
            "Speaker": t.speaker,
            "Text": t.text,
            "Anchor Distance": t.anchor_distance,
            "Anchor Similarity": t.anchor_similarity,
            "Stepwise Distance": t.stepwise_distance,
            "Context Similarity": t.context_similarity,
            "Drift Velocity": t.drift_velocity,
            "Drift Acceleration": t.drift_acceleration,
            "Status": t.drift_status.value,
            "Inflection": t.is_inflection_point,
            "Cluster": t.cluster_id,
            "Keywords": ", ".join(t.keywords),
        })
    return pd.DataFrame(records)
