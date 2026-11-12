"""
Main DriftAnalyzer pipeline orchestrator.
"""

from typing import List, Dict, Optional, Any, Union
import logging
import numpy as np

from .models import (
    Turn,
    TurnMetrics,
    Conversation,
    ConversationReport,
    DriftStatus,
    DriftZone,
)
from .embedders.base import BaseEmbedder
from .embedders.factory import get_embedder
from .metrics.semantic_distance import semantic_distance, cosine_similarity
from .metrics.coherence import (
    compute_turn_to_turn_distances,
    compute_context_window_coherence,
    compute_conversation_cohesion_index,
)
from .metrics.velocity import (
    compute_drift_kinematics,
    classify_drift_status,
    detect_inflection_points,
    identify_drift_zones,
)
from .metrics.topic_tracker import (
    extract_keywords_per_turn,
    cluster_dialogue_turns,
)

logger = logging.getLogger("conversational_drift.analyzer")


class DriftAnalyzer:
    """
    End-to-End NLP Pipeline for Conversational Drift Detection and Coherence Analysis.
    """

    def __init__(
        self,
        embedder: Optional[Union[BaseEmbedder, str]] = "auto",
        model_name: str = "all-MiniLM-L6-v2",
        mild_threshold: float = 0.35,
        moderate_threshold: float = 0.60,
        severe_threshold: float = 0.80,
        velocity_threshold: float = 0.18,
        context_window_size: int = 3,
        device: Optional[str] = None,
    ):
        """
        Initialize the DriftAnalyzer.

        Args:
            embedder: BaseEmbedder instance or string ('auto', 'sentence-transformers', 'tfidf').
            model_name: Model name for sentence-transformers.
            mild_threshold: Semantic distance cutoff for MILD_ELABORATION.
            moderate_threshold: Semantic distance cutoff for MODERATE_DRIFT.
            severe_threshold: Semantic distance cutoff for SEVERE_DRIFT.
            velocity_threshold: Single-turn velocity surge threshold for inflection detection.
            context_window_size: Number of previous turns in moving context window.
            device: 'cpu' or 'cuda' for neural embeddings.
        """
        if isinstance(embedder, str):
            self.embedder = get_embedder(embedder_type=embedder, model_name=model_name, device=device)
        else:
            self.embedder = embedder

        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold
        self.velocity_threshold = velocity_threshold
        self.context_window_size = context_window_size

    def analyze(self, conversation: Conversation) -> ConversationReport:
        """
        Executes full drift analysis on a Conversation instance.

        Args:
            conversation: Conversation object containing dialogue turns.

        Returns:
            ConversationReport containing turn-by-turn metrics, drift zones, and summary.
        """
        if not conversation.turns:
            raise ValueError("Conversation must contain at least one turn for analysis.")

        turns_data = conversation.turns
        turn_texts = [t.text for t in turns_data]
        n_turns = len(turns_data)

        # Determine anchor
        if conversation.anchor_goal and conversation.anchor_goal.strip():
            anchor_text = conversation.anchor_goal.strip()
            anchor_is_turn_zero = False
            all_texts_to_embed = [anchor_text] + turn_texts
        else:
            anchor_text = turn_texts[0]
            anchor_is_turn_zero = True
            all_texts_to_embed = turn_texts

        # 1. Embeddings
        all_embeddings = self.embedder.embed(all_texts_to_embed)

        if anchor_is_turn_zero:
            anchor_emb = all_embeddings[0]
            turn_embeddings = all_embeddings
        else:
            anchor_emb = all_embeddings[0]
            turn_embeddings = all_embeddings[1:]

        # 2. Anchor Similarities & Distances
        anchor_sims = cosine_similarity(turn_embeddings, anchor_emb)
        anchor_dists = np.maximum(0.0, 1.0 - anchor_sims)

        # 3. Step-wise Turn-to-Turn Local Coherence
        step_sims, step_dists = compute_turn_to_turn_distances(turn_embeddings)

        # 4. Context Window Coherence
        context_sims = compute_context_window_coherence(
            turn_embeddings,
            window_size=self.context_window_size
        )

        # 5. Drift Kinematics (Velocity & Acceleration)
        velocities, accelerations = compute_drift_kinematics(anchor_dists)

        # 6. Status Classification & Inflection Points
        statuses = [
            classify_drift_status(
                d,
                mild_threshold=self.mild_threshold,
                moderate_threshold=self.moderate_threshold,
                severe_threshold=self.severe_threshold
            )
            for d in anchor_dists
        ]

        inflection_flags = detect_inflection_points(
            anchor_dists,
            velocities,
            distance_threshold=self.mild_threshold,
            velocity_threshold=self.velocity_threshold
        )

        # 7. Keywords and Topic Clusters
        keywords_per_turn = extract_keywords_per_turn(turn_texts, top_n=4)
        cluster_ids, cluster_summaries = cluster_dialogue_turns(turn_embeddings, turn_texts)

        # Build TurnMetrics objects
        turn_metrics_list: List[TurnMetrics] = []
        for i in range(n_turns):
            tm = TurnMetrics(
                turn_index=turns_data[i].turn_index,
                speaker=turns_data[i].speaker,
                text=turns_data[i].text,
                anchor_similarity=float(round(anchor_sims[i], 4)),
                anchor_distance=float(round(anchor_dists[i], 4)),
                stepwise_similarity=float(round(step_sims[i], 4)),
                stepwise_distance=float(round(step_dists[i], 4)),
                context_similarity=float(round(context_sims[i], 4)),
                drift_velocity=float(round(velocities[i], 4)),
                drift_acceleration=float(round(accelerations[i], 4)),
                drift_status=statuses[i],
                is_inflection_point=bool(inflection_flags[i]),
                keywords=keywords_per_turn[i],
                cluster_id=cluster_ids[i],
                cluster_label=", ".join(cluster_summaries.get(cluster_ids[i], []))
            )
            turn_metrics_list.append(tm)

        # 8. Contiguous Drift Zones
        drift_zones = identify_drift_zones(anchor_dists, statuses, turn_texts)
        # Populate dominant keywords for each zone
        for zone in drift_zones:
            zone_kws: List[str] = []
            for idx in range(zone.start_turn, zone.end_turn + 1):
                zone_kws.extend(keywords_per_turn[idx])
            # unique while preserving order
            seen = set()
            zone.dominant_keywords = [k for k in zone_kws if not (k in seen or seen.add(k))][:5]

        # 9. Global Summary Statistics
        cohesion_idx = compute_conversation_cohesion_index(anchor_dists, step_dists)
        mean_anchor_dist = float(round(float(np.mean(anchor_dists)), 4))
        max_idx = int(np.argmax(anchor_dists))
        max_drift_val = float(round(float(anchor_dists[max_idx]), 4))

        # First inflection turn
        first_inflection = None
        for i, flag in enumerate(inflection_flags):
            if flag and i > 0:
                first_inflection = i
                break

        # 10. Natural Language Synthesis
        exec_summary = self._synthesize_executive_summary(
            conversation_id=conversation.conversation_id,
            total_turns=n_turns,
            cohesion_idx=cohesion_idx,
            mean_dist=mean_anchor_dist,
            first_inflection=first_inflection,
            drift_zones=drift_zones,
            turns=turn_metrics_list,
            anchor_text=anchor_text
        )

        return ConversationReport(
            conversation_id=conversation.conversation_id,
            total_turns=n_turns,
            anchor_text=anchor_text,
            cohesion_index=cohesion_idx,
            mean_anchor_distance=mean_anchor_dist,
            max_drift_turn=max_idx,
            max_drift_distance=max_drift_val,
            first_inflection_turn=first_inflection,
            drift_zones=drift_zones,
            turns=turn_metrics_list,
            cluster_summaries=cluster_summaries,
            executive_summary=exec_summary
        )

    def analyze_turns(
        self,
        turns: List[Dict[str, str]],
        conversation_id: str = "conv_001",
        anchor_goal: Optional[str] = None
    ) -> ConversationReport:
        """
        Convenience method to analyze a list of turn dictionaries.
        Each dict must contain 'speaker' and 'text'.
        """
        conv = Conversation(conversation_id=conversation_id, anchor_goal=anchor_goal)
        for i, t in enumerate(turns):
            conv.add_turn(speaker=t.get("speaker", f"Speaker_{i}"), text=t.get("text", ""))
        return self.analyze(conv)

    def analyze_transcript(
        self,
        transcript_text: str,
        conversation_id: str = "transcript_session",
        anchor_goal: Optional[str] = None
    ) -> ConversationReport:
        """
        Parses raw text dialogues in standard formats (e.g. 'User: ...\\nAgent: ...')
        and runs analysis.
        """
        from .utils import parse_raw_transcript
        conv = parse_raw_transcript(transcript_text, conversation_id=conversation_id, anchor_goal=anchor_goal)
        return self.analyze(conv)

    def _synthesize_executive_summary(
        self,
        conversation_id: str,
        total_turns: int,
        cohesion_idx: float,
        mean_dist: float,
        first_inflection: Optional[int],
        drift_zones: List[DriftZone],
        turns: List[TurnMetrics],
        anchor_text: str
    ) -> str:
        """Generates an intuitive, professional analytical summary."""
        truncated_anchor = anchor_text[:85] + ("..." if len(anchor_text) > 85 else "")

        if cohesion_idx >= 75.0:
            assessment = "Demonstrated exemplary cohesion with minimal topic drift."
        elif cohesion_idx >= 50.0:
            assessment = "Maintained moderate coherence with noticeable topical elaborations."
        else:
            assessment = "Suffered from pronounced semantic drift and topical divergence."

        summary_parts = [
            f"Conversation '{conversation_id}' comprised {total_turns} turns with an overall Cohesion Index of {cohesion_idx}%. {assessment}",
            f"The primary conversational anchor was defined as: \"{truncated_anchor}\"."
        ]

        if first_inflection is not None:
            diverging_turn = turns[first_inflection]
            kws = ", ".join(diverging_turn.keywords[:3]) if diverging_turn.keywords else "emergent context"
            summary_parts.append(
                f"A pivotal semantic inflection occurred at Turn {first_inflection} ({diverging_turn.speaker}), "
                f"where semantic distance accelerated to {diverging_turn.anchor_distance:.3f} centered around [{kws}]."
            )
        else:
            summary_parts.append("No sudden inflection anomalies were detected across the dialogue.")

        if drift_zones:
            zone_desc = "; ".join(
                f"Turns {z.start_turn}–{z.end_turn} ({z.severity.value}, topics: {', '.join(z.dominant_keywords[:3]) or 'general'})"
                for z in drift_zones
            )
            summary_parts.append(f"Identified {len(drift_zones)} distinct persistent drift zone(s): {zone_desc}.")
        else:
            summary_parts.append("The discussion remained tightly bound within permissible topic bounds without persistent drift zones.")

        return " ".join(summary_parts)
