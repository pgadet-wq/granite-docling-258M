"""
MEL/MMEL Document Parser using Granite-Docling-258M.

This module provides the main parser class for extracting structured data
from MEL (Master Equipment List) and MMEL (Master Minimum Equipment List)
aviation documents.
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from .models import (
    MELDocument,
    MELChapter,
    MELItem,
    RepairCategory,
    DispatchCondition,
    OperationalProcedure,
    MaintenanceProcedure,
)
from .extractor import TableExtractor, TextExtractor, DocTagParser, ExtractedTable

logger = logging.getLogger(__name__)


@dataclass
class ParserConfig:
    """Configuration for the MEL/MMEL parser"""
    model_path: str = "."
    device: str = "auto"  # "auto", "cuda", "cpu"
    max_tokens: int = 4096
    batch_size: int = 1
    detect_aircraft_type: bool = True
    detect_document_type: bool = True
    extract_procedures: bool = True
    verbose: bool = False


class MELMMELParser:
    """
    Parser for MEL/MMEL aviation documents using Granite-Docling-258M.

    This parser extracts structured data from MEL/MMEL document images,
    including:
    - ATA chapter organization
    - Equipment items with dispatch requirements
    - Repair categories and intervals
    - Operational and maintenance procedures

    Example usage:
        parser = MELMMELParser()
        parser.load_model("./granite-docling-258M")

        # Parse a single page
        doc = parser.parse_image("mel_page.png")

        # Parse a multi-page PDF
        doc = parser.parse_pdf("mel_document.pdf")

        # Export to JSON
        doc.save_json("parsed_mel.json")
    """

    # Patterns for MEL/MMEL content extraction
    PATTERNS = {
        # ATA Chapter pattern: "21", "21-00", "ATA 21", "Chapter 21"
        "ata_chapter": re.compile(
            r"(?:ATA\s*|Chapter\s*)?(\d{2})(?:-(\d{2}))?",
            re.IGNORECASE
        ),
        # Item number: "1", "1-1", "1.1", "(a)", etc.
        "item_number": re.compile(
            r"^(\d+(?:[-\.]\d+)?|\([a-z]\))",
            re.IGNORECASE
        ),
        # Repair category: single letter A, B, C, D or dash
        "repair_category": re.compile(r"^([ABCD-])$"),
        # Quantity pattern: "2 / 1" or "2/1" or "2 1"
        "quantity": re.compile(r"(\d+)\s*[/\s]\s*(\d+)"),
        # Procedure markers
        "operational_proc": re.compile(r"\(O\)\s*(.*?)(?=\(M\)|$)", re.DOTALL),
        "maintenance_proc": re.compile(r"\(M\)\s*(.*?)(?=\(O\)|$)", re.DOTALL),
        # Aircraft type patterns
        "aircraft_type": re.compile(
            r"(A3[0-9]{2}|A350|B7[0-9]{2}|B737|B747|B757|B767|B777|B787|"
            r"E1[0-9]{2}|E2[0-9]{2}|CRJ|ATR|DHC|MD-[0-9]+|DC-[0-9]+)",
            re.IGNORECASE
        ),
        # Document type
        "document_type": re.compile(
            r"\b(M{1,2}EL|Master\s*(?:Minimum\s*)?Equipment\s*List)\b",
            re.IGNORECASE
        ),
    }

    # Standard MEL table column headers
    MEL_HEADERS = [
        "item", "system", "description", "rectification",
        "number installed", "number required", "remarks",
        "repair interval", "category", "procedures"
    ]

    def __init__(self, config: Optional[ParserConfig] = None):
        """
        Initialize the MEL/MMEL parser.

        Args:
            config: Parser configuration options
        """
        self.config = config or ParserConfig()
        self.table_extractor = TableExtractor()
        self.text_extractor = TextExtractor()
        self._model_loaded = False

    def load_model(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Load the Granite-Docling model.

        Args:
            model_path: Path to model directory (overrides config)
            device: Device to use (overrides config)
        """
        path = model_path or self.config.model_path
        dev = device or self.config.device

        logger.info(f"Loading Granite-Docling model from {path}")

        # Load for both extractors (they share the model)
        self.table_extractor.load_model(path, dev)

        # Share the loaded model
        self.text_extractor.model = self.table_extractor.model
        self.text_extractor.processor = self.table_extractor.processor
        self.text_extractor._model_loaded = True

        self._model_loaded = True
        logger.info("Model loaded successfully")

    def parse_image(
        self,
        image_path: Union[str, Path],
        document: Optional[MELDocument] = None
    ) -> MELDocument:
        """
        Parse a single MEL/MMEL page image.

        Args:
            image_path: Path to the image file
            document: Existing document to append to (for multi-page)

        Returns:
            MELDocument with extracted data
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image_path = Path(image_path)
        logger.info(f"Parsing image: {image_path}")

        # Create new document if not provided
        if document is None:
            document = MELDocument(
                document_type="MEL",
                aircraft_type="Unknown",
                source_file=str(image_path)
            )

        # Extract tables (main MEL content is typically in tables)
        tables = self.table_extractor.extract_tables_from_image(
            str(image_path),
            "Convert table to OTSL."
        )

        # Extract text for headers and metadata
        text_elements = self.text_extractor.extract_text_from_image(
            str(image_path),
            "Convert this page to docling."
        )

        # Process extracted content
        self._process_page_content(document, tables, text_elements)

        return document

    def parse_images(
        self,
        image_paths: List[Union[str, Path]],
        **kwargs
    ) -> MELDocument:
        """
        Parse multiple page images into a single document.

        Args:
            image_paths: List of image file paths
            **kwargs: Additional arguments for document creation

        Returns:
            Complete MELDocument
        """
        document = MELDocument(
            document_type=kwargs.get("document_type", "MEL"),
            aircraft_type=kwargs.get("aircraft_type", "Unknown"),
            operator=kwargs.get("operator"),
            authority=kwargs.get("authority"),
        )

        for path in image_paths:
            self.parse_image(path, document)

        return document

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        dpi: int = 150
    ) -> MELDocument:
        """
        Parse a PDF document.

        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to parse (None = all)
            dpi: Resolution for PDF rendering

        Returns:
            MELDocument with extracted data
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError(
                "pdf2image is required for PDF parsing. "
                "Install with: pip install pdf2image"
            )

        pdf_path = Path(pdf_path)
        logger.info(f"Parsing PDF: {pdf_path}")

        # Convert PDF to images
        if pages:
            images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                first_page=min(pages),
                last_page=max(pages)
            )
        else:
            images = convert_from_path(str(pdf_path), dpi=dpi)

        document = MELDocument(
            document_type="MEL",
            aircraft_type="Unknown",
            source_file=str(pdf_path)
        )

        for i, image in enumerate(images):
            logger.info(f"Processing page {i + 1}/{len(images)}")

            # Extract tables
            tables = self.table_extractor.extract_tables_from_pil(
                image,
                "Convert table to OTSL."
            )

            # Extract text
            text_elements = self.text_extractor.extract_text_from_pil(
                image,
                "Convert this page to docling."
            )

            self._process_page_content(document, tables, text_elements)

        return document

    def _process_page_content(
        self,
        document: MELDocument,
        tables: List[ExtractedTable],
        text_elements: List
    ) -> None:
        """
        Process extracted content and update document.

        Args:
            document: Document to update
            tables: Extracted tables
            text_elements: Extracted text elements
        """
        # Extract metadata from text
        self._extract_metadata(document, text_elements)

        # Process MEL tables
        for table in tables:
            self._process_mel_table(document, table)

    def _extract_metadata(self, document: MELDocument, text_elements: List) -> None:
        """Extract document metadata from text elements."""
        for element in text_elements:
            text = element.content

            # Detect document type
            if self.config.detect_document_type:
                doc_match = self.PATTERNS["document_type"].search(text)
                if doc_match:
                    matched = doc_match.group(1).upper()
                    if "MMEL" in matched or "MINIMUM" in matched:
                        document.document_type = "MMEL"
                    else:
                        document.document_type = "MEL"

            # Detect aircraft type
            if self.config.detect_aircraft_type:
                aircraft_match = self.PATTERNS["aircraft_type"].search(text)
                if aircraft_match:
                    document.aircraft_type = aircraft_match.group(1).upper()

    def _process_mel_table(self, document: MELDocument, table: ExtractedTable) -> None:
        """
        Process a MEL table and extract items.

        Args:
            document: Document to update
            table: Extracted table
        """
        if not table.rows:
            return

        # Identify column mapping
        column_map = self._identify_columns(table.headers)

        if not column_map:
            logger.warning("Could not identify MEL table structure")
            return

        current_chapter = None

        for row in table.rows:
            # Check if this is a chapter header row
            chapter_info = self._extract_chapter_from_row(row)
            if chapter_info:
                chapter_num, chapter_title = chapter_info
                current_chapter = self._get_or_create_chapter(
                    document, chapter_num, chapter_title
                )
                continue

            # Extract item from row
            item = self._extract_item_from_row(row, column_map)
            if item and current_chapter:
                current_chapter.items.append(item)

    def _identify_columns(self, headers: List[str]) -> Dict[str, int]:
        """
        Identify column indices from headers.

        Args:
            headers: Table header row

        Returns:
            Dictionary mapping column types to indices
        """
        column_map = {}
        header_lower = [h.lower().strip() for h in headers]

        # Define header patterns for each column type
        patterns = {
            "item": ["item", "no", "number", "#"],
            "description": ["description", "system", "equipment", "title"],
            "installed": ["installed", "inst", "qty installed"],
            "required": ["required", "req", "dispatch", "qty required"],
            "category": ["category", "cat", "repair", "interval"],
            "remarks": ["remarks", "notes", "conditions", "limitations"],
        }

        for col_type, keywords in patterns.items():
            for i, header in enumerate(header_lower):
                if any(kw in header for kw in keywords):
                    column_map[col_type] = i
                    break

        return column_map

    def _extract_chapter_from_row(self, row: List[str]) -> Optional[tuple]:
        """
        Check if row is a chapter header.

        Args:
            row: Table row

        Returns:
            Tuple of (chapter_number, chapter_title) or None
        """
        row_text = " ".join(row).strip()

        # Check for ATA chapter pattern
        match = self.PATTERNS["ata_chapter"].search(row_text)
        if match:
            chapter_num = match.group(1)
            if match.group(2):
                chapter_num += f"-{match.group(2)}"

            # Get title from standard ATA chapters or from row
            title = MELChapter.get_chapter_title(chapter_num)

            # Try to extract title from row if it's there
            remaining = row_text[match.end():].strip()
            if remaining and len(remaining) > 3:
                # Clean up common separators
                title = re.sub(r"^[-–:.\s]+", "", remaining).strip()

            return (chapter_num, title)

        return None

    def _extract_item_from_row(
        self,
        row: List[str],
        column_map: Dict[str, int]
    ) -> Optional[MELItem]:
        """
        Extract MEL item from table row.

        Args:
            row: Table row data
            column_map: Column type to index mapping

        Returns:
            MELItem or None
        """
        if not row or all(not cell.strip() for cell in row):
            return None

        def get_cell(col_type: str, default: str = "") -> str:
            idx = column_map.get(col_type)
            if idx is not None and idx < len(row):
                return row[idx].strip()
            return default

        # Extract basic fields
        item_number = get_cell("item")
        description = get_cell("description")

        if not item_number and not description:
            return None

        # Parse quantities
        installed = 0
        required = 0

        installed_text = get_cell("installed")
        required_text = get_cell("required")

        if installed_text:
            try:
                installed = int(re.search(r"\d+", installed_text).group())
            except (AttributeError, ValueError):
                pass

        if required_text:
            try:
                required = int(re.search(r"\d+", required_text).group())
            except (AttributeError, ValueError):
                pass

        # Parse repair category
        category_text = get_cell("category", "-")
        category = RepairCategory.NONE
        for cat in RepairCategory:
            if cat.value == category_text.upper():
                category = cat
                break

        # Get remarks
        remarks = get_cell("remarks")

        # Extract procedures from remarks if enabled
        op_procs = []
        maint_procs = []

        if self.config.extract_procedures and remarks:
            op_procs = self._extract_operational_procedures(remarks)
            maint_procs = self._extract_maintenance_procedures(remarks)

        # Determine dispatch condition
        dispatch = DispatchCondition.GO
        if required > 0:
            dispatch = DispatchCondition.GO_IF
        if required >= installed and required > 0:
            dispatch = DispatchCondition.NO_GO

        return MELItem(
            item_number=item_number or "unknown",
            description=description,
            number_installed=installed,
            number_required=required,
            repair_category=category,
            remarks=remarks,
            operational_procedures=op_procs,
            maintenance_procedures=maint_procs,
            dispatch_condition=dispatch,
            raw_text=" | ".join(row)
        )

    def _extract_operational_procedures(self, text: str) -> List[OperationalProcedure]:
        """Extract (O) operational procedures from text."""
        procedures = []

        matches = self.PATTERNS["operational_proc"].findall(text)
        for i, match in enumerate(matches):
            if match.strip():
                proc = OperationalProcedure(
                    procedure_id=f"O-{i+1}",
                    description=match.strip(),
                    steps=self._split_procedure_steps(match)
                )
                procedures.append(proc)

        return procedures

    def _extract_maintenance_procedures(self, text: str) -> List[MaintenanceProcedure]:
        """Extract (M) maintenance procedures from text."""
        procedures = []

        matches = self.PATTERNS["maintenance_proc"].findall(text)
        for i, match in enumerate(matches):
            if match.strip():
                proc = MaintenanceProcedure(
                    procedure_id=f"M-{i+1}",
                    description=match.strip(),
                    steps=self._split_procedure_steps(match)
                )
                procedures.append(proc)

        return procedures

    def _split_procedure_steps(self, text: str) -> List[str]:
        """Split procedure text into individual steps."""
        # Split on numbered items or bullet points
        steps = re.split(r"(?:^|\n)\s*(?:\d+[.)]\s*|[-•]\s*)", text)
        return [s.strip() for s in steps if s.strip()]

    def _get_or_create_chapter(
        self,
        document: MELDocument,
        chapter_number: str,
        title: str
    ) -> MELChapter:
        """Get existing chapter or create new one."""
        # Check if chapter exists
        for chapter in document.chapters:
            if chapter.chapter_number == chapter_number:
                return chapter

        # Create new chapter
        chapter = MELChapter(
            chapter_number=chapter_number,
            title=title
        )
        document.chapters.append(chapter)

        return chapter

    def parse_raw_text(self, text: str) -> MELDocument:
        """
        Parse MEL/MMEL content from raw text (for testing or pre-extracted content).

        Args:
            text: Raw text content

        Returns:
            MELDocument
        """
        document = MELDocument(
            document_type="MEL",
            aircraft_type="Unknown"
        )

        # Simple line-by-line parsing
        lines = text.strip().split("\n")
        current_chapter = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for chapter header
            chapter_info = self._extract_chapter_from_row([line])
            if chapter_info:
                chapter_num, chapter_title = chapter_info
                current_chapter = self._get_or_create_chapter(
                    document, chapter_num, chapter_title
                )
                continue

            # Try to parse as item (simple format)
            parts = re.split(r"\s{2,}|\t", line)
            if len(parts) >= 2 and current_chapter:
                item = MELItem(
                    item_number=parts[0],
                    description=parts[1] if len(parts) > 1 else "",
                    remarks=" ".join(parts[2:]) if len(parts) > 2 else ""
                )
                current_chapter.items.append(item)

        return document


# Convenience function
def parse_mel_document(
    source: Union[str, Path, List[str]],
    model_path: str = ".",
    **kwargs
) -> MELDocument:
    """
    Convenience function to parse MEL/MMEL documents.

    Args:
        source: Image path, PDF path, or list of image paths
        model_path: Path to Granite-Docling model
        **kwargs: Additional arguments for parser

    Returns:
        Parsed MELDocument
    """
    parser = MELMMELParser()
    parser.load_model(model_path)

    if isinstance(source, list):
        return parser.parse_images(source, **kwargs)
    elif str(source).lower().endswith(".pdf"):
        return parser.parse_pdf(source, **kwargs)
    else:
        return parser.parse_image(source)
