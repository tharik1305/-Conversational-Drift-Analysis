# Conversational Drift Analyzer Using NLP 💬📉

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io)

An end-to-end, production-grade Natural Language Processing (NLP) framework designed to measure, detect, alert, and visualize **Conversational Drift** in multi-turn dialogues, customer support transcripts, and LLM chat interactions.

---

## 📌 Table of Contents
- [What is Conversational Drift?](#what-is-conversational-drift)
- [Mathematical & NLP Foundations](#mathematical--nlp-foundations)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Project Directory Structure](#project-directory-structure)
- [Quickstart Guide](#quickstart-guide)
  - [1. Installation](#1-installation)
  - [2. Run the Demo](#2-run-the-demo)
  - [3. Launch the Interactive Web Dashboard](#3-launch-the-interactive-web-dashboard)
  - [4. Start the REST API](#4-start-the-rest-api)
  - [5. Command-Line Interface (CLI)](#5-command-line-interface-cli)
- [Python SDK Usage](#python-sdk-usage)
- [REST API Endpoints](#rest-api-endpoints)
- [Benchmark Datasets](#benchmark-datasets)
- [Testing](#testing)

---

## 🔍 What is Conversational Drift?

In human-to-human conversations and human-AI dialogues, **conversational drift** is the phenomenon where turns gradually or abruptly divert from the original goal, user intent, or core topic.

### Drift Manifestations:
1. **Goal Misalignment**: A customer support conversation intended to resolve an account lockout spirals into off-topic pleasantries (weather, tourist recommendations).
2. **LLM Hallucinatory Deviation**: An AI agent tasked with explaining physics principles progressively wanders into science-fiction speculation or ungrounded tangents.
3. **Local Incoherence**: Abrupt non-sequiturs or disjointed responses between adjacent conversational turns.

The **Conversational Drift Analyzer** quantifies these dynamics into actionable metrics, inflection alerts, and visual trajectories.

---

## 📐 Mathematical & NLP Foundations

### 1. Vector Embeddings
Each turn $t \in \{0, \dots, N\}$ is mapped to a unit-normalized vector $e_t \in \mathbb{R}^D$ such that $\|e_t\|_2 = 1$:
- **Dense Transformer**: HuggingFace `sentence-transformers` (e.g. `all-MiniLM-L6-v2`)
- **Lightweight Fallback**: Sublinear TF-IDF with Latent Semantic Analysis (LSA via TruncatedSVD)

### 2. Anchor Semantic Distance
Let $e_0$ denote the embedding of the conversation's initial prompt or explicit anchor goal $G$. The cumulative anchor distance at turn $t$ is defined as:
$$D_{\text{anchor}}(t) = 1 - (e_t \cdot e_0)$$
where $D_{\text{anchor}}(t) \in [0.0, 2.0]$.

### 3. Step-wise Local Distance
Measures local coherence between consecutive turns $T_t$ and $T_{t-1}$:
$$D_{\text{step}}(t) = 1 - (e_t \cdot e_{t-1})$$

### 4. Sliding Window Context Coherence
Measures alignment with a moving exponentially-decayed centroid of the prior $k$ turns:
$$\bar{e}_{t, k} = \text{normalize}\left( \sum_{j=1}^k \gamma^{k - j} e_{t-j} \right), \quad S_{\text{context}}(t) = e_t \cdot \bar{e}_{t, k}$$

### 5. Drift Kinematics: Velocity & Acceleration
- **Drift Velocity**: First difference of anchor distance:
  $$V(t) = D_{\text{anchor}}(t) - D_{\text{anchor}}(t-1)$$
- **Drift Acceleration**: Second difference:
  $$A(t) = V(t) - V(t-1)$$

### 6. Inflection Point Breakaway Detection
A turn $t^*$ is flagged as an **Inflection Point** if:
$$V(t^*) \ge \theta_{\text{velocity}} \quad \text{or} \quad \left( D_{\text{anchor}}(t^*) \ge \theta_{\text{drift}} \land V(t^*) \ge 0.10 \land D_{\text{anchor}}(t^*-1) < \theta_{\text{drift}} \right)$$

### 7. Conversation Cohesion Index (CCI)
A normalized global health score from $0\%$ to $100\%$:
$$\text{CCI} = \max\left(0, \min\left(100, 100 \times \left(1 - \left[0.60 \bar{D}_{\text{anchor}} + 0.40 \bar{D}_{\text{step}}\right]\right)\right)\right)$$

---

## 🏛️ System Architecture

```
                               ┌────────────────────────────────────────┐
                               │       Multi-Turn Dialogue Input        │
                               │  (JSON / CSV / Raw Text / Live Stream) │
                               └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │            Embedder Engine             │
                               │  • Sentence-Transformers (Dense 384d)  │
                               │  • TF-IDF + LSA SVD (Zero-Torch Mode)  │
                               └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
                ┌──────────────────────────────────┴──────────────────────────────────┐
                │                                                                     │
                ▼                                                                     ▼
┌───────────────────────────────┐                                     ┌───────────────────────────────┐
│     Semantic Drift Engine     │                                     │     Topic & Cluster Engine    │
│ • Anchor Distance             │                                     │ • Turn-Level Keyword TF-IDF   │
│ • Step-wise Local Distance    │                                     │ • KMeans Dialogue Clustering  │
│ • Velocity & Acceleration     │                                     │ • 2D PCA Dimensionality Red.  │
│ • Inflection Point Detection  │                                     └───────────────┬───────────────┘
│ • Contiguous Drift Zones      │                                                     │
└───────────────┬───────────────┘                                                     │
                │                                                                     │
                └──────────────────────────────────┬──────────────────────────────────┘
                                                   │
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │           DriftAnalyzer Core           │
                               │    Executive Natural Language Report   │
                               │    Conversation Cohesion Index (CCI)   │
                               └───────────────────┬────────────────────┘
                                                   │
                ┌──────────────────────────────────┼──────────────────────────────────┐
                ▼                                  ▼                                  ▼
┌───────────────────────────────┐  ┌───────────────────────────────┐  ┌───────────────────────────────┐
│    Streamlit Web Dashboard    │  │       FastAPI REST API        │  │     CLI & Report Exporters    │
│  • Dynamic Plotly Trajectory  │  │  • POST /api/v1/analyze       │  │  • Standalone HTML Dashboard  │
│  • 2D PCA Spatial Motion Map  │  │  • POST /api/v1/quick-check   │  │  • GitHub Markdown Report     │
│  • Live Chat Simulator        │  │  • Full OpenAPI / Swagger     │  │  • Structured JSON Metrics    │
└───────────────────────────────┘  └───────────────────────────────┘  └───────────────────────────────┘
```

---

## 🚀 Key Features

- **Dual-Mode Embedding Pipeline**: Works with high-accuracy dense neural transformers (`sentence-transformers`) and automatically falls back to lightweight, zero-PyTorch TF-IDF/LSA.
- **Dynamic Drift Trajectory Charts**: Interactive Plotly visualizations highlighting drift thresholds (On-Topic, Mild, Moderate, Severe) and breakaway inflection markers.
- **2D Semantic Space Mapping**: PCA projection maps dialogue navigation across semantic space.
- **Executive Natural Language Synthesis**: Automatically produces concise human-readable diagnoses explaining when and why the topic diverged.
- **Contiguous Drift Zone Tracking**: Isolates multi-turn segments that drifted and summarizes dominant keywords.
- **Multiple Interfaces**:
  - **Streamlit Web Dashboard**: Visual analytics, real-time simulator, and export manager.
  - **FastAPI Microservice**: High-throughput REST API for production chat agents.
  - **CLI Tool**: Instant terminal auditing and report generation.

---

## 📂 Project Directory Structure

```
NLP project/
├── README.md                          # Comprehensive documentation & theory
├── requirements.txt                   # Project dependencies
├── setup.py                           # Package installer configuration
├── run_demo.py                        # Standalone demo script
├── run_demo.bat                       # One-click demo launcher (Windows)
├── run_dashboard.bat                  # One-click Streamlit dashboard launcher
├── run_api.bat                        # One-click FastAPI REST server launcher
│
├── conversational_drift/              # Core Library Package
│   ├── __init__.py                    # Public API exports
│   ├── models.py                      # Pydantic schemas (Turn, Conversation, Metrics, Report)
│   ├── analyzer.py                    # Main DriftAnalyzer pipeline
│   ├── reporter.py                    # HTML, Markdown, and JSON exporters
│   ├── utils.py                       # Transcript parsing & data loaders
│   ├── cli.py                         # Command-line entrypoint
│   │
│   ├── embedders/                     # Pluggable Embedding Backends
│   │   ├── base.py                    # BaseEmbedder abstract class
│   │   ├── sentence_transformer.py    # Dense transformer embedder
│   │   ├── tfidf_embedder.py          # Fast TF-IDF + LSA embedder
│   │   └── factory.py                 # Embedder factory with auto-fallback
│   │
│   └── metrics/                       # NLP Metric Calculators
│       ├── semantic_distance.py       # Cosine & Euclidean distances
│       ├── coherence.py               # Local step-wise & sliding context coherence
│       ├── velocity.py                # Kinematics & inflection point detection
│       └── topic_tracker.py           # Keyword extraction, KMeans, & 2D PCA
│
├── app/                               # Interactive Streamlit Web App
│   ├── app.py                         # Streamlit application
│   ├── styles.css                     # Custom styling
│   └── components/                    # Visualization components
│       ├── charts.py                  # Plotly trajectory & 2D PCA charts
│       └── tables.py                  # Formatted metric tables & badges
│
├── api/                               # FastAPI REST Microservice
│   ├── __init__.py
│   └── main.py                        # REST endpoints (/analyze, /quick-check, /health)
│
├── data/                              # Benchmark Datasets
│   ├── customer_support_drift.json    # Off-topic drift into weather/travel
│   ├── customer_support_anchored.json # Tightly focused billing resolution
│   ├── llm_hallucination_drift.json   # Quantum physics to mythological conspiracy
│   └── student_tutoring_drift.json    # Calculus derivatives drifting to gaming
│
└── tests/                             # Test Suite
    ├── test_embedders.py
    ├── test_metrics.py
    ├── test_analyzer.py
    └── test_api.py
```

---

## ⚡ Quickstart Guide

### 1. Installation

Clone or navigate to the workspace directory:
```bash
cd "NLP project"
```

Create a virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On Linux/macOS

pip install -r requirements.txt
```

*(Optional) Install in editable mode:*
```bash
pip install -e .
```

---

### 2. Run the Demo

Run the demo script to analyze a benchmark dialogue and export an interactive HTML report:
```bash
python run_demo.py
# Or double-click run_demo.bat on Windows
```

---

### 3. Launch the Interactive Web Dashboard

Launch the Streamlit web dashboard:
```bash
streamlit run app/app.py
# Or double-click run_dashboard.bat on Windows
```
The dashboard opens at `http://localhost:8501`. Features include:
- Selecting benchmark datasets or uploading your own transcripts.
- Interactive chat simulator to test turn-by-turn drift live.
- Trajectory plots, kinematics curves, and 2D semantic space projection.
- One-click downloads for standalone HTML dashboards and Markdown reports.

---

### 4. Start the REST API

Launch the FastAPI microservice:
```bash
uvicorn api.main:app --reload --port 8000
# Or double-click run_api.bat on Windows
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

### 5. Command-Line Interface (CLI)

Analyze a JSON file:
```bash
python -m conversational_drift.cli -f data/customer_support_drift.json --format terminal
```

Analyze a raw text transcript and export to HTML:
```bash
python -m conversational_drift.cli -f transcript.txt -o drift_report.html --format html
```

---

## 🐍 Python SDK Usage

```python
from conversational_drift import DriftAnalyzer, Conversation

# 1. Initialize analyzer
analyzer = DriftAnalyzer(embedder="auto")

# 2. Build or load conversation
conv = Conversation(
    conversation_id="session_01",
    anchor_goal="Resolve customer account authentication error"
)
conv.add_turn("Customer", "I cannot log into my account, it says incorrect password.")
conv.add_turn("Agent", "I can help with that. Can you confirm your registered email address?")
conv.add_turn("Customer", "Sure, it is user@example.com. By the way, do you like Italian food?")
conv.add_turn("Agent", "I love Italian food! Especially lasagna.")

# 3. Analyze
report = analyzer.analyze(conv)

# 4. Inspect results
print(f"Cohesion Index: {report.cohesion_index}%")
print(f"First Inflection Turn: {report.first_inflection_turn}")
print(f"Executive Summary: {report.executive_summary}")

for turn in report.turns:
    print(f"Turn #{turn.turn_index} ({turn.speaker}) -> Dist: {turn.anchor_distance:.3f} | Status: {turn.drift_status.value}")
```

---

## 🌐 REST API Endpoints

### `POST /api/v1/analyze`
Submits a multi-turn dialogue with custom thresholds and returns complete metrics.

**Sample Request Body:**
```json
{
  "conversation_id": "support_ticket_88",
  "anchor_goal": "Troubleshoot WiFi connectivity",
  "turns": [
    {"speaker": "User", "text": "My router is blinking red."},
    {"speaker": "Agent", "text": "Please restart your router by unplugging the power cable."}
  ]
}
```

### `POST /api/v1/quick-check`
Real-time turn audit endpoint suitable for streaming chat agents.

**Sample Request Body:**
```json
{
  "anchor_text": "Troubleshoot router issue",
  "current_turn_text": "What do you think about the latest football match?"
}
```

**Response:**
```json
{
  "anchor_distance": 0.8421,
  "anchor_similarity": 0.1579,
  "is_drifting": true,
  "drift_status": "SEVERE_DRIFT"
}
```

---

## 🧪 Testing

Run the test suite using `pytest`:
```bash
pytest -v
```

All unit and integration tests verify:
- Vector normalization and TF-IDF / LSA embedding stability.
- Mathematical correctness of distance, velocity, and acceleration calculations.
- Turn status classification and inflection breakaway identification.
- End-to-end report generation and FastAPI endpoint contracts.

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
