#!/usr/bin/env python3
"""
Command-line interface for MEL/MMEL Parser.

Usage:
    python -m mel_mmel_parser.cli parse document.pdf -o output.json
    python -m mel_mmel_parser.cli parse image.png --format markdown
    python -m mel_mmel_parser.cli info document.json
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Optional

from .parser import MELMMELParser, ParserConfig
from .models import MELDocument


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_command(args):
    """Handle the parse command."""
    config = ParserConfig(
        model_path=args.model_path,
        device=args.device,
        verbose=args.verbose
    )

    parser = MELMMELParser(config)

    print(f"Loading model from: {args.model_path}")
    parser.load_model()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing: {input_path}")

    # Parse based on file type
    if input_path.suffix.lower() == ".pdf":
        document = parser.parse_pdf(input_path)
    elif input_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        document = parser.parse_image(input_path)
    else:
        print(f"Error: Unsupported file type: {input_path.suffix}", file=sys.stderr)
        sys.exit(1)

    # Output results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(f".{args.format}")

    if args.format == "json":
        document.save_json(str(output_path))
        print(f"Saved JSON output to: {output_path}")
    elif args.format == "markdown":
        md_content = export_to_markdown(document)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Saved Markdown output to: {output_path}")
    elif args.format == "summary":
        summary = document.summary()
        print(json.dumps(summary, indent=2))

    # Print summary
    print(f"\nDocument Summary:")
    print(f"  Type: {document.document_type}")
    print(f"  Aircraft: {document.aircraft_type}")
    print(f"  Chapters: {len(document.chapters)}")
    print(f"  Total Items: {len(document.get_all_items())}")


def info_command(args):
    """Handle the info command."""
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        document = MELDocument.load_json(str(input_path))
    except Exception as e:
        print(f"Error loading document: {e}", file=sys.stderr)
        sys.exit(1)

    summary = document.summary()
    print(json.dumps(summary, indent=2))


def search_command(args):
    """Handle the search command."""
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    document = MELDocument.load_json(str(input_path))
    results = document.search_items(args.keyword)

    print(f"Found {len(results)} items matching '{args.keyword}':\n")

    for item in results:
        print(f"  [{item.item_number}] {item.description}")
        print(f"    Category: {item.repair_category.value}")
        print(f"    Installed/Required: {item.number_installed}/{item.number_required}")
        if item.remarks:
            print(f"    Remarks: {item.remarks[:100]}...")
        print()


def export_to_markdown(document: MELDocument) -> str:
    """Export document to Markdown format."""
    lines = []

    lines.append(f"# {document.document_type} - {document.aircraft_type}")
    lines.append("")

    if document.operator:
        lines.append(f"**Operator:** {document.operator}")
    if document.authority:
        lines.append(f"**Authority:** {document.authority}")
    if document.revision_number:
        lines.append(f"**Revision:** {document.revision_number}")
    if document.effective_date:
        lines.append(f"**Effective Date:** {document.effective_date}")

    lines.append("")
    lines.append("---")
    lines.append("")

    for chapter in document.chapters:
        lines.append(f"## Chapter {chapter.chapter_number}: {chapter.title}")
        lines.append("")

        if chapter.items:
            # Table header
            lines.append("| Item | Description | Inst | Req | Cat | Remarks |")
            lines.append("|------|-------------|------|-----|-----|---------|")

            for item in chapter.items:
                remarks = item.remarks[:50] + "..." if len(item.remarks) > 50 else item.remarks
                remarks = remarks.replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| {item.item_number} | {item.description} | "
                    f"{item.number_installed} | {item.number_required} | "
                    f"{item.repair_category.value} | {remarks} |"
                )

            lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="mel-mmel-parser",
        description="MEL/MMEL Document Parser using Granite-Docling-258M"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse a MEL/MMEL document")
    parse_parser.add_argument("input", help="Input file (PDF or image)")
    parse_parser.add_argument(
        "-o", "--output",
        help="Output file path"
    )
    parse_parser.add_argument(
        "-f", "--format",
        choices=["json", "markdown", "summary"],
        default="json",
        help="Output format (default: json)"
    )
    parse_parser.add_argument(
        "-m", "--model-path",
        default=".",
        help="Path to Granite-Docling model"
    )
    parse_parser.add_argument(
        "-d", "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use (default: auto)"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show document info")
    info_parser.add_argument("input", help="JSON document file")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search items in document")
    search_parser.add_argument("input", help="JSON document file")
    search_parser.add_argument("keyword", help="Keyword to search")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "parse":
        parse_command(args)
    elif args.command == "info":
        info_command(args)
    elif args.command == "search":
        search_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
