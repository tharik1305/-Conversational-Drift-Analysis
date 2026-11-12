"""
Command-line interface (CLI) for the Conversational Drift Analyzer.
"""

import sys
import argparse
from pathlib import Path
from .analyzer import DriftAnalyzer
from .utils import load_conversation_from_json, parse_raw_transcript
from .reporter import DriftReporter


def main():
    parser = argparse.ArgumentParser(
        description="Conversational Drift Analyzer - Measure & Detect Topical Divergence in Dialogues"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        required=True,
        help="Path to conversation file (.json or .txt raw transcript)"
    )
    parser.add_argument(
        "-a", "--anchor",
        type=str,
        default=None,
        help="Explicit anchor goal or initial query (optional, defaults to turn 0)"
    )
    parser.add_argument(
        "-e", "--embedder",
        type=str,
        default="auto",
        choices=["auto", "sentence-transformers", "tfidf"],
        help="Embedding backend ('auto', 'sentence-transformers', 'tfidf')"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (e.g. report.html, report.md, or report.json)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="terminal",
        choices=["terminal", "html", "md", "json"],
        help="Report output format"
    )

    args = parser.parse_args()
    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading conversation from {file_path}...")
    if file_path.suffix.lower() == ".json":
        conversation = load_conversation_from_json(str(file_path))
        if args.anchor:
            conversation.anchor_goal = args.anchor
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        conversation = parse_raw_transcript(text, conversation_id=file_path.stem, anchor_goal=args.anchor)

    print(f"Analyzing {len(conversation.turns)} turns using embedder='{args.embedder}'...")
    analyzer = DriftAnalyzer(embedder=args.embedder)
    report = analyzer.analyze(conversation)

    # Format or save output
    if args.output:
        out_fmt = args.format if args.format != "terminal" else Path(args.output).suffix.lstrip(".") or "html"
        saved = DriftReporter.save_report(report, args.output, format_type=out_fmt)
        print(f"Report saved to: {saved}")
    else:
        if args.format == "terminal" or args.format == "md":
            print("\n" + DriftReporter.to_markdown(report) + "\n")
        elif args.format == "json":
            print(DriftReporter.to_json(report))
        elif args.format == "html":
            print(DriftReporter.to_html(report))


if __name__ == "__main__":
    main()
