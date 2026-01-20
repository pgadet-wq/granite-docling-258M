"""
Text and Table extraction utilities using Granite-Docling-258M.

This module provides low-level extraction capabilities for processing
document images using the Granite-Docling vision-language model.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TableCell:
    """Represents a single cell in a table"""
    row: int
    col: int
    content: str
    rowspan: int = 1
    colspan: int = 1


@dataclass
class ExtractedTable:
    """Represents an extracted table from the document"""
    rows: List[List[str]] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    raw_otsl: str = ""  # Original OTSL format from Granite-Docling

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headers": self.headers,
            "rows": self.rows,
            "raw_otsl": self.raw_otsl
        }

    def get_column(self, index: int) -> List[str]:
        """Get all values in a column"""
        return [row[index] if index < len(row) else "" for row in self.rows]

    def get_row(self, index: int) -> List[str]:
        """Get all values in a row"""
        return self.rows[index] if index < len(self.rows) else []


@dataclass
class ExtractedText:
    """Represents extracted text with structure information"""
    content: str
    element_type: str  # paragraph, header, list_item, etc.
    confidence: float = 1.0
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2


class DocTagParser:
    """
    Parser for Granite-Docling DocTags format.

    DocTags is an XML-like markup format used by Granite-Docling to represent
    structured document content.
    """

    # DocTag patterns
    DOCTAG_PATTERNS = {
        "text": re.compile(r"<text>(.*?)</text>", re.DOTALL),
        "paragraph": re.compile(r"<paragraph>(.*?)</paragraph>", re.DOTALL),
        "section_header": re.compile(r"<section_header_level_(\d+)>(.*?)</section_header_level_\d+>", re.DOTALL),
        "table": re.compile(r"<otsl>(.*?)</otsl>", re.DOTALL),
        "list_item": re.compile(r"<list_item>(.*?)</list_item>", re.DOTALL),
        "formula": re.compile(r"<formula>(.*?)</formula>", re.DOTALL),
        "code": re.compile(r"<code>(.*?)</code>", re.DOTALL),
        "page_header": re.compile(r"<page_header>(.*?)</page_header>", re.DOTALL),
        "page_footer": re.compile(r"<page_footer>(.*?)</page_footer>", re.DOTALL),
    }

    # Table cell patterns for OTSL format
    CELL_PATTERNS = {
        "fcel": re.compile(r"<fcel>(.*?)</fcel>", re.DOTALL),  # First cell
        "ecel": re.compile(r"<ecel>(.*?)</ecel>", re.DOTALL),  # End cell
        "lcel": re.compile(r"<lcel>(.*?)</lcel>", re.DOTALL),  # Left cell
        "ucel": re.compile(r"<ucel>(.*?)</ucel>", re.DOTALL),  # Upper cell
        "xcel": re.compile(r"<xcel>(.*?)</xcel>", re.DOTALL),  # Cross cell
        "nl": re.compile(r"<nl\s*/>"),  # New line in table
    }

    @classmethod
    def parse_doctags(cls, doctag_output: str) -> Dict[str, List[str]]:
        """
        Parse DocTags output into structured elements.

        Args:
            doctag_output: Raw DocTags string from Granite-Docling

        Returns:
            Dictionary with element types as keys and lists of content as values
        """
        result = {key: [] for key in cls.DOCTAG_PATTERNS.keys()}

        for element_type, pattern in cls.DOCTAG_PATTERNS.items():
            matches = pattern.findall(doctag_output)
            if element_type == "section_header":
                # Include level information for headers
                result[element_type] = [(level, text.strip()) for level, text in matches]
            else:
                result[element_type] = [m.strip() for m in matches]

        return result

    @classmethod
    def parse_otsl_table(cls, otsl_content: str) -> ExtractedTable:
        """
        Parse OTSL (Open Table Structure Language) format into structured table.

        OTSL uses special tokens:
        - <fcel>: First cell (starts a new row)
        - <ecel>: End cell (ends a row)
        - <lcel>: Left merged cell
        - <ucel>: Upper merged cell
        - <xcel>: Cross merged cell (both left and upper)
        - <nl />: New line within table

        Args:
            otsl_content: OTSL formatted table string

        Returns:
            ExtractedTable with parsed rows and headers
        """
        table = ExtractedTable(raw_otsl=otsl_content)

        # Split by newlines first
        lines = otsl_content.split("<nl")

        current_row = []
        all_rows = []

        # Pattern to match all cell types
        cell_pattern = re.compile(r"<(fcel|ecel|lcel|ucel|xcel)>(.*?)</\1>", re.DOTALL)

        for line in lines:
            matches = cell_pattern.findall(line)
            for cell_type, content in matches:
                content = content.strip()

                if cell_type == "fcel":
                    # First cell - might start a new row
                    if current_row:
                        all_rows.append(current_row)
                    current_row = [content]
                elif cell_type == "ecel":
                    # End cell
                    current_row.append(content)
                    all_rows.append(current_row)
                    current_row = []
                elif cell_type in ("lcel", "ucel", "xcel"):
                    # Merged cells - use empty or reference
                    current_row.append(content if content else "")
                else:
                    current_row.append(content)

        # Don't forget the last row
        if current_row:
            all_rows.append(current_row)

        # First row is typically headers
        if all_rows:
            table.headers = all_rows[0]
            table.rows = all_rows[1:] if len(all_rows) > 1 else []

        return table

    @classmethod
    def extract_plain_text(cls, doctag_output: str) -> str:
        """
        Extract plain text from DocTags, removing all markup.

        Args:
            doctag_output: Raw DocTags string

        Returns:
            Clean text without tags
        """
        # Remove all tags but keep content
        text = re.sub(r"<[^>]+>", " ", doctag_output)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class TableExtractor:
    """
    Extracts and processes tables from document images using Granite-Docling.
    """

    def __init__(self, model=None, processor=None):
        """
        Initialize TableExtractor.

        Args:
            model: Loaded Granite-Docling model (optional, will load if needed)
            processor: Loaded processor (optional)
        """
        self.model = model
        self.processor = processor
        self._model_loaded = model is not None

    def load_model(self, model_path: str = ".", device: str = "auto"):
        """
        Load the Granite-Docling model.

        Args:
            model_path: Path to model directory
            device: Device to load model on ("auto", "cuda", "cpu")
        """
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading Granite-Docling model from {model_path} on {device}")

            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map=device if device == "cuda" else None
            )

            if device == "cpu":
                self.model = self.model.to(device)

            self._model_loaded = True
            logger.info("Model loaded successfully")

        except ImportError as e:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            ) from e

    def extract_tables_from_image(
        self,
        image_path: str,
        instruction: str = "Convert table to OTSL."
    ) -> List[ExtractedTable]:
        """
        Extract tables from a document image.

        Args:
            image_path: Path to the image file
            instruction: Extraction instruction for the model

        Returns:
            List of extracted tables
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        return self.extract_tables_from_pil(image, instruction)

    def extract_tables_from_pil(
        self,
        image,
        instruction: str = "Convert table to OTSL."
    ) -> List[ExtractedTable]:
        """
        Extract tables from a PIL Image.

        Args:
            image: PIL Image object
            instruction: Extraction instruction

        Returns:
            List of extracted tables
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch

        # Prepare the prompt
        prompt = f"<|start_of_role|>user<|end_of_role|><image>{instruction}<|end_of_text|><|start_of_role|>assistant<|end_of_role|>"

        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )

        # Decode output
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=False)

        # Extract tables from DocTags output
        tables = []
        parsed = DocTagParser.parse_doctags(generated_text)

        for otsl_content in parsed.get("table", []):
            table = DocTagParser.parse_otsl_table(otsl_content)
            tables.append(table)

        return tables


class TextExtractor:
    """
    Extracts structured text from document images using Granite-Docling.
    """

    def __init__(self, model=None, processor=None):
        """
        Initialize TextExtractor.

        Args:
            model: Loaded Granite-Docling model
            processor: Loaded processor
        """
        self.model = model
        self.processor = processor
        self._model_loaded = model is not None

    def load_model(self, model_path: str = ".", device: str = "auto"):
        """Load the Granite-Docling model."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map=device if device == "cuda" else None
            )

            if device == "cpu":
                self.model = self.model.to(device)

            self._model_loaded = True

        except ImportError as e:
            raise ImportError(
                "transformers and torch are required."
            ) from e

    def extract_text_from_image(
        self,
        image_path: str,
        instruction: str = "Convert this page to docling."
    ) -> List[ExtractedText]:
        """
        Extract structured text from a document image.

        Args:
            image_path: Path to image file
            instruction: Extraction instruction

        Returns:
            List of extracted text elements
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        return self.extract_text_from_pil(image, instruction)

    def extract_text_from_pil(
        self,
        image,
        instruction: str = "Convert this page to docling."
    ) -> List[ExtractedText]:
        """
        Extract structured text from a PIL Image.

        Args:
            image: PIL Image
            instruction: Extraction instruction

        Returns:
            List of ExtractedText objects
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch

        prompt = f"<|start_of_role|>user<|end_of_role|><image>{instruction}<|end_of_text|><|start_of_role|>assistant<|end_of_role|>"

        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )

        generated_text = self.processor.decode(outputs[0], skip_special_tokens=False)

        # Parse into structured elements
        return self._parse_to_extracted_text(generated_text)

    def _parse_to_extracted_text(self, doctag_output: str) -> List[ExtractedText]:
        """Parse DocTags output into ExtractedText objects."""
        results = []
        parsed = DocTagParser.parse_doctags(doctag_output)

        # Process paragraphs
        for content in parsed.get("paragraph", []):
            results.append(ExtractedText(
                content=content,
                element_type="paragraph"
            ))

        # Process text blocks
        for content in parsed.get("text", []):
            results.append(ExtractedText(
                content=content,
                element_type="text"
            ))

        # Process headers
        for level, content in parsed.get("section_header", []):
            results.append(ExtractedText(
                content=content,
                element_type=f"header_level_{level}"
            ))

        # Process list items
        for content in parsed.get("list_item", []):
            results.append(ExtractedText(
                content=content,
                element_type="list_item"
            ))

        return results

    def get_raw_doctags(
        self,
        image_path: str,
        instruction: str = "Convert this page to docling."
    ) -> str:
        """
        Get raw DocTags output from the model.

        Args:
            image_path: Path to image
            instruction: Extraction instruction

        Returns:
            Raw DocTags string
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")

        prompt = f"<|start_of_role|>user<|end_of_role|><image>{instruction}<|end_of_text|><|start_of_role|>assistant<|end_of_role|>"

        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )

        return self.processor.decode(outputs[0], skip_special_tokens=False)
