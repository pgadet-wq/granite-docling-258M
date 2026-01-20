"""
MEL/MMEL Parser using Granite-Docling-258M

A specialized parser for Master Equipment List (MEL) and Master Minimum Equipment List (MMEL)
aviation documents using IBM's Granite-Docling vision-language model.

Author: Generated with Granite-Docling-258M
License: Apache 2.0
"""

from .models import (
    MELItem,
    MELChapter,
    MELDocument,
    DispatchCondition,
    OperationalProcedure,
    MaintenanceProcedure,
)
from .parser import MELMMELParser
from .extractor import TableExtractor, TextExtractor

__version__ = "0.1.0"
__all__ = [
    "MELItem",
    "MELChapter",
    "MELDocument",
    "DispatchCondition",
    "OperationalProcedure",
    "MaintenanceProcedure",
    "MELMMELParser",
    "TableExtractor",
    "TextExtractor",
]
