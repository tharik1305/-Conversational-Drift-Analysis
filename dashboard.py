"""
Conversational Drift Analyzer - Interactive Streamlit Web Dashboard
Run with:
    python -m streamlit run dashboard.py
"""

import sys
from pathlib import Path
import json
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conversational_drift import (
    DriftAnalyzer,
    Conversation,
    Turn,
    DriftReporter,
    parse_raw_transcript,
    load_conversation_from_json,
    report_to_dataframe,
)
from conversational_drift.metrics.topic_tracker import compute_2d_trajectory_coords
from app.components.charts import (
    create_drift_trajectory_chart,
    create_kinematics_chart,
    create_2d_trajectory_chart,
)
from app.components.tables import build_dataframe_for_ui

# Page Config
st.set_page_config(
    page_title="Conversational Drift Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional CSS for card polish
css_file = PROJECT_ROOT / "app" / "styles.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.title("⚙️ Engine Configuration")

embedder_choice = st.sidebar.selectbox(
    "Embedding Backend",
    options=["auto", "sentence-transformers", "tfidf"],
    index=0,
    help="'auto' uses sentence-transformers if installed, otherwise fast TF-IDF/LSA."
)

model_name = st.sidebar.text_input(
    "Transformer Model",
    value="all-MiniLM-L6-v2",
    help="HuggingFace model identifier (e.g. all-MiniLM-L6-v2)"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Drift Sensitivity Thresholds")

col_th1, col_th2 = st.sidebar.columns(2)
with col_th1:
    mild_th = st.slider("Mild Drift Cutoff", 0.10, 0.50, 0.35, 0.05)
    severe_th = st.slider("Severe Drift Cutoff", 0.60, 1.20, 0.80, 0.05)
with col_th2:
    mod_th = st.slider("Moderate Cutoff", 0.30, 0.80, 0.60, 0.05)
    vel_th = st.slider("Velocity Jump", 0.05, 0.40, 0.18, 0.02)

context_win = st.sidebar.slider("Context Window Size (k turns)", 1, 6, 3)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Conversational Drift** measures how a dialogue progressively deviates "
    "from its anchor topic or explicit goal across turns using dense semantic embeddings."
)

# ----------------- MAIN INTERFACE -----------------
st.title("Conversational Drift Analyzer 💬📉")
st.markdown(
    "Analyze multi-turn dialogues to detect **topic divergence, local incoherence, inflection points, and persistent drift zones**."
)

# Input tabs
input_mode = st.radio(
    "Choose Input Method:",
    options=["Benchmark Datasets", "Paste Transcript", "Upload File", "Interactive Chat Simulator"],
    horizontal=True
)

conversation_to_analyze = None
custom_anchor = None

if input_mode == "Benchmark Datasets":
    sample_options = {
        "Customer Support (Drift to Weather/Seattle)": "customer_support_drift.json",
        "Customer Support (Anchored Billing Issue)": "customer_support_anchored.json",
        "LLM Generation (Quantum to Atlantis Hallucination)": "llm_hallucination_drift.json",
        "Student Tutoring (Calculus to Video Games)": "student_tutoring_drift.json",
    }
    selected_sample = st.selectbox("Select Benchmark Dialogue:", list(sample_options.keys()))
    sample_path = PROJECT_ROOT / "data" / sample_options[selected_sample]

    if sample_path.exists():
        conversation_to_analyze = load_conversation_from_json(str(sample_path))
        with st.expander("👁️ View Benchmark Dialogue Turns", expanded=False):
            st.json(conversation_to_analyze.model_dump())

elif input_mode == "Paste Transcript":
    raw_text = st.text_area(
        "Paste Dialogue (Format: 'Speaker: text' per line)",
        value="""Customer: I need help resetting my account password.
Agent: Sure! Have you received a verification code on your email?
Customer: Yes, but the link says expired.
Agent: I just sent a new pin to your inbox.
Customer: Thanks! Say, what's your favorite video game lately?
Agent: I really love playing Elden Ring on the weekends!
Customer: Me too! What build do you use for your character?
Agent: A strength and faith hybrid build with a colossal sword.""",
        height=200
    )
    custom_anchor = st.text_input("Explicit Anchor / Goal (Optional):", value="")
    if raw_text.strip():
        conversation_to_analyze = parse_raw_transcript(
            raw_text,
            conversation_id="user_pasted_transcript",
            anchor_goal=custom_anchor.strip() if custom_anchor.strip() else None
        )

elif input_mode == "Upload File":
    uploaded = st.file_uploader("Upload JSON or TXT Conversation", type=["json", "txt"])
    custom_anchor = st.text_input("Explicit Anchor / Goal (Optional):", value="")
    if uploaded is not None:
        if uploaded.name.endswith(".json"):
            data = json.load(uploaded)
            temp_path = PROJECT_ROOT / "data" / "_temp_upload.json"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            conversation_to_analyze = load_conversation_from_json(str(temp_path))
            if custom_anchor.strip():
                conversation_to_analyze.anchor_goal = custom_anchor.strip()
            if temp_path.exists():
                temp_path.unlink()
        else:
            text = uploaded.read().decode("utf-8")
            conversation_to_analyze = parse_raw_transcript(
                text,
                conversation_id=uploaded.name,
                anchor_goal=custom_anchor.strip() if custom_anchor.strip() else None
            )

elif input_mode == "Interactive Chat Simulator":
    st.markdown("Build a conversation turn-by-turn to test real-time drift sensitivity:")
    if "sim_turns" not in st.session_state:
        st.session_state.sim_turns = [
            {"speaker": "User", "text": "Can you recommend a machine learning book for beginners?"},
            {"speaker": "Bot", "text": "I recommend 'Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow' by Aurélien Géron."},
        ]

    sim_anchor = st.text_input("Conversation Goal Anchor:", value="Find recommendations for machine learning textbooks")

    col_s1, col_s2 = st.columns([1, 4])
    with col_s1:
        new_speaker = st.selectbox("Speaker", ["User", "Bot", "Agent", "Customer"])
    with col_s2:
        new_text = st.text_input("Turn Text", placeholder="Type message here...")

    col_b1, col_b2 = st.columns([1, 4])
    with col_b1:
        if st.button("➕ Add Turn") and new_text.strip():
            st.session_state.sim_turns.append({"speaker": new_speaker, "text": new_text.strip()})
            st.rerun()
    with col_b2:
        if st.button("🗑️ Reset Turns"):
            st.session_state.sim_turns = []
            st.rerun()

    # Display current turns
    for idx, t in enumerate(st.session_state.sim_turns):
        st.caption(f"**Turn {idx} ({t['speaker']}):** {t['text']}")

    if st.session_state.sim_turns:
        conversation_to_analyze = Conversation(
            conversation_id="simulator_session",
            anchor_goal=sim_anchor.strip() if sim_anchor.strip() else None
        )
        for i, t in enumerate(st.session_state.sim_turns):
            conversation_to_analyze.add_turn(t["speaker"], t["text"])

# ----------------- ANALYSIS TRIGGER -----------------
st.markdown("")
trigger_col1, trigger_col2 = st.columns([1, 3])
with trigger_col1:
    run_btn = st.button("🚀 Analyze Conversational Drift", type="primary", use_container_width=True)

if run_btn:
    if not conversation_to_analyze or not conversation_to_analyze.turns:
        st.error("Please provide or select a valid conversation with at least 2 turns.")
    else:
        with st.spinner("Analyzing semantic drift, kinematics, and topic clusters..."):
            analyzer = DriftAnalyzer(
                embedder=embedder_choice,
                model_name=model_name,
                mild_threshold=mild_th,
                moderate_threshold=mod_th,
                severe_threshold=severe_th,
                velocity_threshold=vel_th,
                context_window_size=context_win
            )
            report = analyzer.analyze(conversation_to_analyze)
            st.session_state["report"] = report

            turn_texts = [t.text for t in conversation_to_analyze.turns]
            embs = analyzer.embedder.embed(turn_texts)
            st.session_state["coords_2d"] = compute_2d_trajectory_coords(embs)

# ----------------- REPORT DISPLAY -----------------
if "report" in st.session_state:
    report = st.session_state["report"]
    coords_2d = st.session_state.get("coords_2d")

    st.markdown("---")
    st.subheader(f"📊 Analysis Results: `{report.conversation_id}`")

    # KPI Summary Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            "Cohesion Index",
            f"{report.cohesion_index:.1f}%",
            delta="Healthy" if report.cohesion_index >= 70 else ("Degraded" if report.cohesion_index >= 50 else "Severe Drift"),
            delta_color="normal" if report.cohesion_index >= 70 else "inverse"
        )
    with m2:
        st.metric("Total Turns", report.total_turns)
    with m3:
        st.metric("Mean Distance", f"{report.mean_anchor_distance:.3f}")
    with m4:
        st.metric("Peak Drift Turn", f"Turn #{report.max_drift_turn}")
    with m5:
        first_inf = f"Turn #{report.first_inflection_turn}" if report.first_inflection_turn is not None else "None"
        st.metric("First Inflection", first_inf)

    # Status Alert Banner
    if report.first_inflection_turn is not None:
        st.markdown(
            f'<div class="custom-alert danger">⚠️ <b>Inflection Point Detected:</b> The dialogue experienced an abrupt '
            f'divergence at <b>Turn #{report.first_inflection_turn}</b> initiated by '
            f'<b>{report.turns[report.first_inflection_turn].speaker}</b>.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="custom-alert success">✅ <b>Smooth Trajectory:</b> No abrupt inflection breakaways were detected.</div>',
            unsafe_allow_html=True
        )

    # Executive Summary Box
    st.markdown(
        f'<div class="summary-container"><b>Executive NLP Insights:</b><br>{report.executive_summary}</div>',
        unsafe_allow_html=True
    )

    # Visualization Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Drift Trajectory",
        "🗺️ 2D Semantic Space",
        "🔍 Turn-by-Turn Inspector",
        "🏷️ Topic Clusters & Keywords",
        "📥 Export & Reports"
    ])

    with tab1:
        st.plotly_chart(create_drift_trajectory_chart(report), use_container_width=True)
        with st.expander("📊 View Drift Kinematics (Velocity & Acceleration Subplots)", expanded=False):
            st.plotly_chart(create_kinematics_chart(report), use_container_width=True)

    with tab2:
        st.markdown(
            "This 2D PCA projection maps the conversation's trajectory across semantic space. "
            "A cohesive conversation forms a tight cluster; a drifting conversation travels far across the plane."
        )
        if coords_2d is not None:
            st.plotly_chart(create_2d_trajectory_chart(report, coords_2d), use_container_width=True)

    with tab3:
        st.markdown("### Dialogue Utterance Breakdown")
        df = build_dataframe_for_ui(report.turns)
        st.dataframe(df, use_container_width=True)

        st.markdown("#### Turn Cards Details")
        for t in report.turns:
            badge_icon = "🟢" if t.drift_status.value == "ON_TOPIC" else ("🔵" if t.drift_status.value == "MILD_ELABORATION" else ("🟡" if t.drift_status.value == "MODERATE_DRIFT" else "🔴"))
            alert_tag = " [⚠️ INFLECTION]" if t.is_inflection_point else ""
            with st.expander(f"{badge_icon} Turn #{t.turn_index} — {t.speaker}: \"{t.text[:60]}...\"{alert_tag}"):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Anchor Distance:** `{t.anchor_distance:.4f}`")
                c1.write(f"**Step-wise Distance:** `{t.stepwise_distance:.4f}`")
                c2.write(f"**Velocity:** `{t.drift_velocity:+.4f}`")
                c2.write(f"**Status:** `{t.drift_status.value}`")
                c3.write(f"**Cluster ID:** `{t.cluster_id}` ({t.cluster_label})")
                c3.write(f"**Salient Terms:** `{', '.join(t.keywords)}`")
                st.info(f"**Full Text:** {t.text}")

    with tab4:
        st.markdown("### Topic Clusters Identified in Dialogue")
        if report.cluster_summaries:
            col_k1, col_k2 = st.columns(2)
            for c_id, terms in report.cluster_summaries.items():
                turns_in_c = [t.turn_index for t in report.turns if t.cluster_id == c_id]
                with col_k1 if c_id % 2 == 0 else col_k2:
                    st.markdown(
                        f"""
                        <div style="background:rgba(30,41,59,0.7); border:1px solid rgba(148,163,184,0.2); padding:16px; border-radius:10px; margin-bottom:12px;">
                            <h4 style="margin:0 0 8px 0; color:#38bdf8;">Topic Cluster #{c_id}</h4>
                            <p style="margin:0 0 6px 0; font-size:14px;"><b>Top Keyphrases:</b> {', '.join(terms)}</p>
                            <p style="margin:0; font-size:12px; color:#94a3b8;"><b>Associated Turns:</b> {turns_in_c}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    with tab5:
        st.markdown("### Export Analysis Report")
        col_ex1, col_ex2, col_ex3 = st.columns(3)

        html_content = DriftReporter.to_html(report)
        md_content = DriftReporter.to_markdown(report)
        json_content = DriftReporter.to_json(report)

        with col_ex1:
            st.download_button(
                "📥 Download HTML Dashboard",
                data=html_content,
                file_name=f"{report.conversation_id}_drift_report.html",
                mime="text/html"
            )
        with col_ex2:
            st.download_button(
                "📥 Download Markdown Report",
                data=md_content,
                file_name=f"{report.conversation_id}_drift_report.md",
                mime="text/markdown"
            )
        with col_ex3:
            st.download_button(
                "📥 Download JSON Data",
                data=json_content,
                file_name=f"{report.conversation_id}_metrics.json",
                mime="application/json"
            )
