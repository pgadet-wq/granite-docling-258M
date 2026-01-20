#!/usr/bin/env python3
"""
Example usage of the MEL/MMEL Parser.

This script demonstrates how to use the parser to extract
structured data from MEL/MMEL aviation documents.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mel_mmel_parser import (
    MELMMELParser,
    MELDocument,
    MELChapter,
    MELItem,
    RepairCategory,
)
from mel_mmel_parser.parser import ParserConfig


def example_parse_image():
    """Example: Parse a single MEL page image."""
    print("=" * 60)
    print("Example 1: Parse a single image")
    print("=" * 60)

    # Initialize parser
    config = ParserConfig(
        model_path="../",  # Path to granite-docling-258M
        device="auto",
        verbose=True
    )
    parser = MELMMELParser(config)

    # Load the model
    print("Loading Granite-Docling model...")
    parser.load_model()

    # Parse an image
    # document = parser.parse_image("mel_page.png")

    # For demo, create a sample document
    document = create_sample_document()

    # Export to JSON
    document.save_json("parsed_mel.json")
    print(f"Saved to parsed_mel.json")

    # Print summary
    print("\nDocument Summary:")
    print(f"  Type: {document.document_type}")
    print(f"  Aircraft: {document.aircraft_type}")
    print(f"  Chapters: {len(document.chapters)}")
    print(f"  Items: {len(document.get_all_items())}")


def example_parse_pdf():
    """Example: Parse a multi-page PDF."""
    print("\n" + "=" * 60)
    print("Example 2: Parse a PDF document")
    print("=" * 60)

    parser = MELMMELParser()
    parser.load_model("../")

    # Parse PDF (requires pdf2image)
    # document = parser.parse_pdf("mel_document.pdf")

    print("PDF parsing requires: pip install pdf2image")
    print("And poppler-utils installed on your system")


def example_search_items():
    """Example: Search for items in a parsed document."""
    print("\n" + "=" * 60)
    print("Example 3: Search items")
    print("=" * 60)

    # Load existing document
    # document = MELDocument.load_json("parsed_mel.json")

    # For demo
    document = create_sample_document()

    # Search by keyword
    results = document.search_items("fire")
    print(f"Found {len(results)} items containing 'fire':")
    for item in results:
        print(f"  - [{item.item_number}] {item.description}")

    # Get items by repair category
    category_b_items = document.get_items_by_category(RepairCategory.B)
    print(f"\nCategory B items: {len(category_b_items)}")


def example_dispatch_check():
    """Example: Check dispatch conditions."""
    print("\n" + "=" * 60)
    print("Example 4: Dispatch check")
    print("=" * 60)

    document = create_sample_document()

    # Check specific item
    item = document.get_item("26", "1")
    if item:
        print(f"Item: {item.description}")
        print(f"  Installed: {item.number_installed}")
        print(f"  Required: {item.number_required}")
        print(f"  Category: {item.repair_category.value}")

        # Check if can dispatch with 1 available
        can_dispatch = item.is_dispatchable(available_quantity=1)
        print(f"  Can dispatch with 1 available: {can_dispatch}")


def create_sample_document() -> MELDocument:
    """Create a sample MEL document for demonstration."""
    document = MELDocument(
        document_type="MEL",
        aircraft_type="A320",
        operator="Sample Airlines",
        authority="EASA",
        revision_number="Rev 15",
        effective_date="2024-01-15"
    )

    # Chapter 26: Fire Protection
    chapter_26 = MELChapter(
        chapter_number="26",
        title="Fire Protection"
    )

    chapter_26.items.append(MELItem(
        item_number="1",
        description="Engine Fire Detection System",
        number_installed=2,
        number_required=2,
        repair_category=RepairCategory.A,
        remarks="Both systems must be operational for dispatch.",
    ))

    chapter_26.items.append(MELItem(
        item_number="2",
        description="APU Fire Detection System",
        number_installed=1,
        number_required=0,
        repair_category=RepairCategory.C,
        remarks="May be inoperative provided APU is not used.",
    ))

    document.chapters.append(chapter_26)

    # Chapter 32: Landing Gear
    chapter_32 = MELChapter(
        chapter_number="32",
        title="Landing Gear"
    )

    chapter_32.items.append(MELItem(
        item_number="1",
        description="Landing Gear Position Indicators",
        number_installed=3,
        number_required=3,
        repair_category=RepairCategory.A,
        remarks="All indicators must be operational.",
    ))

    chapter_32.items.append(MELItem(
        item_number="2",
        description="Nose Wheel Steering",
        number_installed=1,
        number_required=1,
        repair_category=RepairCategory.B,
        remarks="(M) Maintenance procedure required to verify alternate steering available.",
    ))

    document.chapters.append(chapter_32)

    return document


def example_export_formats():
    """Example: Export to different formats."""
    print("\n" + "=" * 60)
    print("Example 5: Export formats")
    print("=" * 60)

    document = create_sample_document()

    # Export to JSON
    json_str = document.to_json(indent=2)
    print("JSON Export (first 500 chars):")
    print(json_str[:500] + "...")

    # Export summary
    print("\nSummary:")
    import json
    print(json.dumps(document.summary(), indent=2))


if __name__ == "__main__":
    print("MEL/MMEL Parser Examples")
    print("Using Granite-Docling-258M\n")

    # Run examples
    example_parse_image()
    example_search_items()
    example_dispatch_check()
    example_export_formats()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
