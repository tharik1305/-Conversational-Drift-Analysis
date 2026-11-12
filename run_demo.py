"""
Quickstart demonstration script for Conversational Drift Analyzer.
Run with: python run_demo.py
"""

from pathlib import Path
from conversational_drift import (
    DriftAnalyzer,
    load_conversation_from_json,
    DriftReporter,
)

def run_demo():
    print("=" * 70)
    print("💬 CONVERSATIONAL DRIFT ANALYZER — NLP DEMO")
    print("=" * 70)

    # 1. Load sample dataset
    sample_path = Path("data") / "customer_support_drift.json"
    print(f"\n[1] Loading benchmark dialogue: '{sample_path}'...")
    conv = load_conversation_from_json(str(sample_path))
    print(f"    Loaded {len(conv.turns)} turns.")
    print(f"    Anchor Goal: \"{conv.anchor_goal}\"")

    # 2. Run Drift Analysis
    print("\n[2] Initializing NLP DriftAnalyzer (auto-detects backend)...")
    analyzer = DriftAnalyzer(embedder="auto")
    print(f"    Active Embedder: {analyzer.embedder.name}")
    print("    Calculating semantic distances, velocity, and inflection points...")
    report = analyzer.analyze(conv)

    # 3. Print Results Summary
    print("\n" + "=" * 70)
    print("📊 ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Overall Cohesion Index : {report.cohesion_index:.1f}%")
    print(f"Mean Semantic Distance : {report.mean_anchor_distance:.3f}")
    print(f"Peak Drift Turn        : Turn #{report.max_drift_turn} (Distance: {report.max_drift_distance:.3f})")
    first_inf = f"Turn #{report.first_inflection_turn}" if report.first_inflection_turn is not None else "None"
    print(f"First Inflection Alert : {first_inf}")
    print(f"\nExecutive Summary:\n> {report.executive_summary}\n")

    # 4. Print Turn-by-Turn Table
    print(f"{'Turn':<6}{'Speaker':<12}{'Anchor Dist':<14}{'Status':<18}{'Utterance'}")
    print("-" * 75)
    for t in report.turns:
        flag = " [🚨 ALERT]" if t.is_inflection_point else ""
        snippet = (t.text[:35] + "...") if len(t.text) > 35 else t.text
        print(f"#{t.turn_index:<5}{t.speaker:<12}{t.anchor_distance:<14.3f}{t.drift_status.value + flag:<18}{snippet}")

    # 5. Export HTML Report
    out_html = "customer_support_drift_report.html"
    DriftReporter.save_report(report, out_html, format_type="html")
    print(f"\n[3] 📄 Interactive HTML Report generated: '{out_html}'")

    print("\n" + "=" * 70)
    print("🚀 HOW TO RUN THE INTERACTIVE DASHBOARD AND API:")
    print("  • Web Dashboard : streamlit run app/app.py")
    print("  • REST API      : uvicorn api.main:app --reload --port 8000")
    print("  • Run Tests     : pytest")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()
