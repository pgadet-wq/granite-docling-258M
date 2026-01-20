"""
Data models for MEL/MMEL documents.

MEL (Master Equipment List) and MMEL (Master Minimum Equipment List) are aviation
documents that define the minimum equipment required for aircraft dispatch.

Structure based on ICAO/FAA/EASA standards:
- ATA Chapters (system classification)
- Items with dispatch conditions
- Operational and maintenance procedures
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import json


class RepairCategory(Enum):
    """MEL Repair Categories (A, B, C, D)"""
    A = "A"  # As specified in remarks (typically 3 calendar days)
    B = "B"  # 3 calendar days
    C = "C"  # 10 calendar days
    D = "D"  # 120 calendar days
    NONE = "-"  # No repair interval specified


class DispatchCondition(Enum):
    """Dispatch conditions for MEL items"""
    GO = "GO"  # May dispatch
    GO_IF = "GO_IF"  # May dispatch with conditions
    NO_GO = "NO_GO"  # May not dispatch


@dataclass
class OperationalProcedure:
    """Operational procedure (O) for flight crew"""
    procedure_id: str
    description: str
    steps: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "procedure_id": self.procedure_id,
            "description": self.description,
            "steps": self.steps,
            "notes": self.notes
        }


@dataclass
class MaintenanceProcedure:
    """Maintenance procedure (M) for ground crew"""
    procedure_id: str
    description: str
    steps: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    estimated_time: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "procedure_id": self.procedure_id,
            "description": self.description,
            "steps": self.steps,
            "required_tools": self.required_tools,
            "estimated_time": self.estimated_time,
            "notes": self.notes
        }


@dataclass
class MELItem:
    """
    Individual MEL/MMEL item representing an equipment or system.

    Attributes:
        item_number: Unique identifier within the chapter (e.g., "1", "1-1", "2")
        description: Equipment/system description
        number_installed: Total quantity installed on aircraft
        number_required: Minimum quantity required for dispatch
        repair_category: A, B, C, or D repair interval
        remarks: Additional conditions and notes
        operational_procedures: List of (O) procedures
        maintenance_procedures: List of (M) procedures
        dispatch_condition: GO, GO_IF, or NO_GO
        exceptions: Any exceptions to standard rules
    """
    item_number: str
    description: str
    number_installed: int = 0
    number_required: int = 0
    repair_category: RepairCategory = RepairCategory.NONE
    remarks: str = ""
    operational_procedures: List[OperationalProcedure] = field(default_factory=list)
    maintenance_procedures: List[MaintenanceProcedure] = field(default_factory=list)
    dispatch_condition: DispatchCondition = DispatchCondition.GO
    exceptions: List[str] = field(default_factory=list)
    sub_items: List["MELItem"] = field(default_factory=list)
    raw_text: str = ""  # Original extracted text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_number": self.item_number,
            "description": self.description,
            "number_installed": self.number_installed,
            "number_required": self.number_required,
            "repair_category": self.repair_category.value,
            "remarks": self.remarks,
            "operational_procedures": [p.to_dict() for p in self.operational_procedures],
            "maintenance_procedures": [p.to_dict() for p in self.maintenance_procedures],
            "dispatch_condition": self.dispatch_condition.value,
            "exceptions": self.exceptions,
            "sub_items": [item.to_dict() for item in self.sub_items]
        }

    def is_dispatchable(self, available_quantity: int) -> bool:
        """Check if aircraft can dispatch with given available quantity"""
        return available_quantity >= self.number_required


@dataclass
class MELChapter:
    """
    ATA Chapter containing MEL items.

    ATA chapters are standardized system classifications:
    - 21: Air Conditioning
    - 22: Auto Flight
    - 23: Communications
    - 24: Electrical Power
    - 25: Equipment/Furnishings
    - 26: Fire Protection
    - 27: Flight Controls
    - 28: Fuel
    - 29: Hydraulic Power
    - 30: Ice and Rain Protection
    - 31: Indicating/Recording Systems
    - 32: Landing Gear
    - 33: Lights
    - 34: Navigation
    - 35: Oxygen
    - 36: Pneumatic
    - 38: Water/Waste
    - 49: Airborne Auxiliary Power
    - 52: Doors
    - 71-80: Powerplant
    """
    chapter_number: str  # ATA chapter (e.g., "21", "22-10")
    title: str
    items: List[MELItem] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    effective_date: Optional[str] = None
    revision: Optional[str] = None

    # Standard ATA chapter titles
    ATA_CHAPTERS = {
        "21": "Air Conditioning",
        "22": "Auto Flight",
        "23": "Communications",
        "24": "Electrical Power",
        "25": "Equipment/Furnishings",
        "26": "Fire Protection",
        "27": "Flight Controls",
        "28": "Fuel",
        "29": "Hydraulic Power",
        "30": "Ice and Rain Protection",
        "31": "Indicating/Recording Systems",
        "32": "Landing Gear",
        "33": "Lights",
        "34": "Navigation",
        "35": "Oxygen",
        "36": "Pneumatic",
        "38": "Water/Waste",
        "49": "Airborne Auxiliary Power",
        "52": "Doors",
        "71": "Powerplant",
        "72": "Engine",
        "73": "Engine Fuel and Control",
        "74": "Ignition",
        "75": "Air",
        "76": "Engine Controls",
        "77": "Engine Indicating",
        "78": "Exhaust",
        "79": "Oil",
        "80": "Starting",
    }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter_number": self.chapter_number,
            "title": self.title,
            "items": [item.to_dict() for item in self.items],
            "notes": self.notes,
            "effective_date": self.effective_date,
            "revision": self.revision
        }

    @classmethod
    def get_chapter_title(cls, chapter_number: str) -> str:
        """Get standard ATA chapter title"""
        base_chapter = chapter_number.split("-")[0]
        return cls.ATA_CHAPTERS.get(base_chapter, f"Chapter {chapter_number}")


@dataclass
class MELDocument:
    """
    Complete MEL/MMEL document.

    Attributes:
        document_type: "MEL" or "MMEL"
        aircraft_type: Aircraft model (e.g., "A320", "B737-800")
        operator: Airline/operator name (for MEL)
        authority: Regulatory authority (e.g., "FAA", "EASA")
        revision_number: Document revision
        effective_date: Date document becomes effective
        chapters: List of ATA chapters with items
    """
    document_type: str  # "MEL" or "MMEL"
    aircraft_type: str
    operator: Optional[str] = None
    authority: Optional[str] = None
    revision_number: Optional[str] = None
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None
    chapters: List[MELChapter] = field(default_factory=list)
    preamble: str = ""
    definitions: Dict[str, str] = field(default_factory=dict)
    general_notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None
    parsed_at: Optional[str] = None

    def __post_init__(self):
        if self.parsed_at is None:
            self.parsed_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "aircraft_type": self.aircraft_type,
            "operator": self.operator,
            "authority": self.authority,
            "revision_number": self.revision_number,
            "effective_date": self.effective_date,
            "expiration_date": self.expiration_date,
            "chapters": [chapter.to_dict() for chapter in self.chapters],
            "preamble": self.preamble,
            "definitions": self.definitions,
            "general_notes": self.general_notes,
            "metadata": self.metadata,
            "source_file": self.source_file,
            "parsed_at": self.parsed_at
        }

    def to_json(self, indent: int = 2) -> str:
        """Export document to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, filepath: str) -> None:
        """Save document to JSON file"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MELDocument":
        """Create MELDocument from dictionary"""
        chapters = []
        for ch_data in data.get("chapters", []):
            items = []
            for item_data in ch_data.get("items", []):
                item = MELItem(
                    item_number=item_data["item_number"],
                    description=item_data["description"],
                    number_installed=item_data.get("number_installed", 0),
                    number_required=item_data.get("number_required", 0),
                    repair_category=RepairCategory(item_data.get("repair_category", "-")),
                    remarks=item_data.get("remarks", ""),
                    dispatch_condition=DispatchCondition(item_data.get("dispatch_condition", "GO")),
                    exceptions=item_data.get("exceptions", [])
                )
                items.append(item)

            chapter = MELChapter(
                chapter_number=ch_data["chapter_number"],
                title=ch_data["title"],
                items=items,
                notes=ch_data.get("notes", []),
                effective_date=ch_data.get("effective_date"),
                revision=ch_data.get("revision")
            )
            chapters.append(chapter)

        return cls(
            document_type=data["document_type"],
            aircraft_type=data["aircraft_type"],
            operator=data.get("operator"),
            authority=data.get("authority"),
            revision_number=data.get("revision_number"),
            effective_date=data.get("effective_date"),
            expiration_date=data.get("expiration_date"),
            chapters=chapters,
            preamble=data.get("preamble", ""),
            definitions=data.get("definitions", {}),
            general_notes=data.get("general_notes", []),
            metadata=data.get("metadata", {}),
            source_file=data.get("source_file"),
            parsed_at=data.get("parsed_at")
        )

    @classmethod
    def from_json(cls, json_str: str) -> "MELDocument":
        """Create MELDocument from JSON string"""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load_json(cls, filepath: str) -> "MELDocument":
        """Load MELDocument from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())

    def get_chapter(self, chapter_number: str) -> Optional[MELChapter]:
        """Get chapter by number"""
        for chapter in self.chapters:
            if chapter.chapter_number == chapter_number:
                return chapter
        return None

    def get_item(self, chapter_number: str, item_number: str) -> Optional[MELItem]:
        """Get specific item by chapter and item number"""
        chapter = self.get_chapter(chapter_number)
        if chapter:
            for item in chapter.items:
                if item.item_number == item_number:
                    return item
        return None

    def get_all_items(self) -> List[MELItem]:
        """Get all items from all chapters"""
        items = []
        for chapter in self.chapters:
            items.extend(chapter.items)
        return items

    def search_items(self, keyword: str) -> List[MELItem]:
        """Search items by keyword in description or remarks"""
        keyword_lower = keyword.lower()
        results = []
        for item in self.get_all_items():
            if (keyword_lower in item.description.lower() or
                keyword_lower in item.remarks.lower()):
                results.append(item)
        return results

    def get_items_by_category(self, category: RepairCategory) -> List[MELItem]:
        """Get all items with specific repair category"""
        return [item for item in self.get_all_items()
                if item.repair_category == category]

    def summary(self) -> Dict[str, Any]:
        """Generate document summary"""
        all_items = self.get_all_items()
        categories = {}
        for cat in RepairCategory:
            count = len([i for i in all_items if i.repair_category == cat])
            if count > 0:
                categories[cat.value] = count

        return {
            "document_type": self.document_type,
            "aircraft_type": self.aircraft_type,
            "total_chapters": len(self.chapters),
            "total_items": len(all_items),
            "items_by_category": categories,
            "chapters": [
                {"number": ch.chapter_number, "title": ch.title, "items": len(ch.items)}
                for ch in self.chapters
            ]
        }
