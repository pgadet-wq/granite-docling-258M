# MEL/MMEL Parser using Granite-Docling-258M

A specialized parser for **Master Equipment List (MEL)** and **Master Minimum Equipment List (MMEL)** aviation documents, powered by IBM's Granite-Docling-258M vision-language model.

## Overview

This parser extracts structured data from MEL/MMEL document images and PDFs, including:

- **ATA Chapter organization** (21-80 system classifications)
- **Equipment items** with dispatch requirements
- **Repair categories** (A, B, C, D intervals)
- **Operational (O) and Maintenance (M) procedures**
- **Dispatch conditions** (GO, GO_IF, NO_GO)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For PDF support
pip install pdf2image
# Also requires poppler-utils: apt-get install poppler-utils

# For GPU acceleration (optional)
pip install accelerate
```

## Quick Start

### Python API

```python
from mel_mmel_parser import MELMMELParser, MELDocument

# Initialize parser
parser = MELMMELParser()
parser.load_model("./")  # Path to granite-docling-258M model

# Parse a document image
document = parser.parse_image("mel_page.png")

# Parse a PDF
document = parser.parse_pdf("mel_document.pdf")

# Export to JSON
document.save_json("parsed_mel.json")

# Search for items
fire_items = document.search_items("fire")
for item in fire_items:
    print(f"[{item.item_number}] {item.description}")

# Get items by repair category
from mel_mmel_parser import RepairCategory
cat_a_items = document.get_items_by_category(RepairCategory.A)
```

### Command Line

```bash
# Parse a PDF to JSON
python -m mel_mmel_parser.cli parse mel_document.pdf -o output.json

# Parse an image to Markdown
python -m mel_mmel_parser.cli parse mel_page.png --format markdown

# Show document info
python -m mel_mmel_parser.cli info output.json

# Search items
python -m mel_mmel_parser.cli search output.json "fire detection"
```

## Data Models

### MELDocument

The top-level container for a MEL/MMEL document:

```python
MELDocument(
    document_type="MEL",      # "MEL" or "MMEL"
    aircraft_type="A320",     # Aircraft model
    operator="Air France",    # Airline (for MEL)
    authority="EASA",         # Regulatory authority
    revision_number="Rev 15",
    effective_date="2024-01-15",
    chapters=[...],           # List of MELChapter
)
```

### MELChapter

ATA chapter containing items:

```python
MELChapter(
    chapter_number="26",
    title="Fire Protection",
    items=[...],  # List of MELItem
)
```

### MELItem

Individual equipment/system entry:

```python
MELItem(
    item_number="1",
    description="Engine Fire Detection System",
    number_installed=2,
    number_required=2,
    repair_category=RepairCategory.A,
    remarks="Both systems must be operational.",
    dispatch_condition=DispatchCondition.GO_IF,
)
```

## Repair Categories

| Category | Interval |
|----------|----------|
| A | As specified in remarks |
| B | 3 calendar days |
| C | 10 calendar days |
| D | 120 calendar days |

## ATA Chapters

Common ATA chapters in MEL/MMEL documents:

| Chapter | System |
|---------|--------|
| 21 | Air Conditioning |
| 22 | Auto Flight |
| 23 | Communications |
| 24 | Electrical Power |
| 26 | Fire Protection |
| 27 | Flight Controls |
| 28 | Fuel |
| 29 | Hydraulic Power |
| 32 | Landing Gear |
| 33 | Lights |
| 34 | Navigation |
| 71-80 | Powerplant |

## Architecture

```
mel_mmel_parser/
├── __init__.py       # Package exports
├── models.py         # Data structures (MELDocument, MELItem, etc.)
├── parser.py         # Main parser using Granite-Docling
├── extractor.py      # Low-level extraction utilities
├── cli.py            # Command-line interface
└── examples/         # Usage examples
```

## How It Works

1. **Image Processing**: Document images are processed by Granite-Docling-258M
2. **Table Extraction**: Tables are converted to OTSL (Open Table Structure Language)
3. **Content Parsing**: DocTags markup is parsed into structured elements
4. **MEL Mapping**: Content is mapped to MEL/MMEL data structures
5. **Export**: Data can be exported to JSON, Markdown, or queried programmatically

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.40+
- Pillow 9.0+
- pdf2image (optional, for PDF support)

## License

Apache 2.0 (same as Granite-Docling-258M)

## Credits

- **Granite-Docling-258M**: IBM Research
- **Docling Framework**: IBM
