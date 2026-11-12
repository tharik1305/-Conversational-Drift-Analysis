"""
FastAPI REST microservice for Conversational Drift Analysis.
Run with: uvicorn api.main:app --reload --port 8000
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conversational_drift import (
    DriftAnalyzer,
    Conversation,
    ConversationReport,
    Turn,
    parse_raw_transcript,
)

app = FastAPI(
    title="Conversational Drift Analyzer API",
    description="REST microservice to detect, measure, and analyze conversational drift in dialogues using NLP.",
    version="1.0.0"
)

# CORS middleware for web integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer cache
_analyzer = None

def get_analyzer() -> DriftAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = DriftAnalyzer(embedder="auto")
    return _analyzer


# Request Schemas
class TurnInput(BaseModel):
    speaker: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class AnalyzeConversationRequest(BaseModel):
    conversation_id: str = "api_session"
    anchor_goal: Optional[str] = None
    turns: List[TurnInput]
    embedder: Optional[str] = "auto"
    mild_threshold: Optional[float] = 0.35
    moderate_threshold: Optional[float] = 0.60
    severe_threshold: Optional[float] = 0.80


class AnalyzeTranscriptRequest(BaseModel):
    conversation_id: str = "transcript_session"
    anchor_goal: Optional[str] = None
    transcript_text: str


class QuickCheckRequest(BaseModel):
    anchor_text: str
    current_turn_text: str
    previous_turn_text: Optional[str] = None


class QuickCheckResponse(BaseModel):
    anchor_distance: float
    anchor_similarity: float
    is_drifting: bool
    drift_status: str
    stepwise_distance: Optional[float] = None


@app.get("/api/v1/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "conversational-drift-analyzer", "version": "1.0.0"}


@app.post("/api/v1/analyze", response_model=ConversationReport)
def analyze_dialogue(req: AnalyzeConversationRequest):
    """
    Analyzes a multi-turn conversation provided as a structured turn list.
    """
    if not req.turns:
        raise HTTPException(status_code=400, detail="Turns list must contain at least one utterance.")

    analyzer = DriftAnalyzer(
        embedder=req.embedder or "auto",
        mild_threshold=req.mild_threshold or 0.35,
        moderate_threshold=req.moderate_threshold or 0.60,
        severe_threshold=req.severe_threshold or 0.80
    )

    conv = Conversation(conversation_id=req.conversation_id, anchor_goal=req.anchor_goal)
    for i, t in enumerate(req.turns):
        conv.turns.append(Turn(
            turn_index=i,
            speaker=t.speaker,
            text=t.text,
            metadata=t.metadata or {}
        ))

    report = analyzer.analyze(conv)
    return report


@app.post("/api/v1/analyze-transcript", response_model=ConversationReport)
def analyze_raw_transcript(req: AnalyzeTranscriptRequest):
    """
    Parses and analyzes an unformatted or standard raw text transcript ('Speaker: utterance').
    """
    if not req.transcript_text.strip():
        raise HTTPException(status_code=400, detail="Transcript text cannot be empty.")

    conv = parse_raw_transcript(
        req.transcript_text,
        conversation_id=req.conversation_id,
        anchor_goal=req.anchor_goal
    )
    analyzer = get_analyzer()
    return analyzer.analyze(conv)


@app.post("/api/v1/quick-check", response_model=QuickCheckResponse)
def quick_turn_check(req: QuickCheckRequest):
    """
    Ultra-fast real-time check for live streaming bots: checks if a single new utterance is drifting.
    """
    analyzer = get_analyzer()
    texts = [req.anchor_text, req.current_turn_text]
    if req.previous_turn_text:
        texts.append(req.previous_turn_text)

    embs = analyzer.embedder.embed(texts)
    anchor_emb = embs[0]
    curr_emb = embs[1]

    sim = float(np.dot(anchor_emb, curr_emb))
    dist = max(0.0, 1.0 - sim)

    step_dist = None
    if req.previous_turn_text and len(embs) > 2:
        prev_emb = embs[2]
        step_sim = float(np.dot(curr_emb, prev_emb))
        step_dist = max(0.0, 1.0 - step_sim)

    from conversational_drift.metrics.velocity import classify_drift_status
    status = classify_drift_status(dist)

    return QuickCheckResponse(
        anchor_distance=round(dist, 4),
        anchor_similarity=round(sim, 4),
        is_drifting=status.value in ("MODERATE_DRIFT", "SEVERE_DRIFT"),
        drift_status=status.value,
        stepwise_distance=round(step_dist, 4) if step_dist is not None else None
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
