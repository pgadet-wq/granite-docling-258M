"""
Web interface for MEL/MMEL Parser.

A Flask application providing a user-friendly interface for parsing
MEL/MMEL aviation documents using Granite-Docling-258M.

Usage:
    python -m mel_mmel_parser.web.app
    # Or
    cd mel_mmel_parser/web && python app.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mel_mmel_parser import MELMMELParser, MELDocument
from mel_mmel_parser.parser import ParserConfig

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# Global parser instance (lazy loaded)
_parser = None
_model_loaded = False


def get_parser():
    """Get or initialize the parser."""
    global _parser, _model_loaded

    if _parser is None:
        _parser = MELMMELParser(ParserConfig(
            model_path=os.environ.get('MODEL_PATH', str(Path(__file__).parent.parent.parent)),
            device=os.environ.get('DEVICE', 'auto')
        ))

    return _parser


def ensure_model_loaded():
    """Ensure model is loaded."""
    global _model_loaded

    parser = get_parser()
    if not _model_loaded:
        parser.load_model()
        _model_loaded = True

    return parser


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Check parser status."""
    return jsonify({
        'status': 'ready',
        'model_loaded': _model_loaded,
        'version': '0.1.0'
    })


@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load the Granite-Docling model."""
    try:
        ensure_model_loaded()
        return jsonify({
            'success': True,
            'message': 'Model loaded successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/parse', methods=['POST'])
def parse_document():
    """Parse an uploaded document."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check file extension
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_extensions:
        return jsonify({
            'error': f'Unsupported file type: {ext}. Allowed: {", ".join(allowed_extensions)}'
        }), 400

    try:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Parse document
        parser = ensure_model_loaded()

        if ext == '.pdf':
            document = parser.parse_pdf(filepath)
        else:
            document = parser.parse_image(filepath)

        # Clean up
        os.remove(filepath)

        return jsonify({
            'success': True,
            'document': document.to_dict(),
            'summary': document.summary()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/parse-demo', methods=['POST'])
def parse_demo():
    """Return a demo document (no model required)."""
    from mel_mmel_parser.models import MELChapter, MELItem, RepairCategory

    document = MELDocument(
        document_type="MEL",
        aircraft_type="A320-200",
        operator="Demo Airlines",
        authority="EASA",
        revision_number="Rev 23",
        effective_date="2024-01-15"
    )

    # Chapter 26: Fire Protection
    ch26 = MELChapter(chapter_number="26", title="Fire Protection")
    ch26.items.append(MELItem(
        item_number="1",
        description="Engine Fire Detection System",
        number_installed=2,
        number_required=2,
        repair_category=RepairCategory.A,
        remarks="Both engine fire detection loops must be operational."
    ))
    ch26.items.append(MELItem(
        item_number="2",
        description="APU Fire Detection System",
        number_installed=1,
        number_required=0,
        repair_category=RepairCategory.C,
        remarks="May be inoperative provided APU is not used in flight."
    ))
    ch26.items.append(MELItem(
        item_number="3",
        description="Cargo Compartment Smoke Detection",
        number_installed=4,
        number_required=4,
        repair_category=RepairCategory.A,
        remarks="All detectors must be operational."
    ))
    document.chapters.append(ch26)

    # Chapter 32: Landing Gear
    ch32 = MELChapter(chapter_number="32", title="Landing Gear")
    ch32.items.append(MELItem(
        item_number="1",
        description="Landing Gear Position Indicators",
        number_installed=3,
        number_required=3,
        repair_category=RepairCategory.A,
        remarks="All three green indicators must illuminate."
    ))
    ch32.items.append(MELItem(
        item_number="2",
        description="Nose Wheel Steering System",
        number_installed=1,
        number_required=1,
        repair_category=RepairCategory.B,
        remarks="(M) Verify alternate steering available."
    ))
    ch32.items.append(MELItem(
        item_number="3",
        description="Brake Wear Indicators",
        number_installed=4,
        number_required=0,
        repair_category=RepairCategory.D,
        remarks="May be inoperative provided brakes are inspected daily."
    ))
    document.chapters.append(ch32)

    # Chapter 34: Navigation
    ch34 = MELChapter(chapter_number="34", title="Navigation")
    ch34.items.append(MELItem(
        item_number="1",
        description="GPS Receivers",
        number_installed=2,
        number_required=1,
        repair_category=RepairCategory.B,
        remarks="One GPS must be operational for RNAV operations."
    ))
    ch34.items.append(MELItem(
        item_number="2",
        description="Weather Radar",
        number_installed=1,
        number_required=0,
        repair_category=RepairCategory.C,
        remarks="May be inoperative for day VFR operations only."
    ))
    document.chapters.append(ch34)

    return jsonify({
        'success': True,
        'document': document.to_dict(),
        'summary': document.summary(),
        'demo': True
    })


@app.route('/api/search', methods=['POST'])
def search_items():
    """Search items in a document."""
    data = request.get_json()

    if not data or 'document' not in data or 'keyword' not in data:
        return jsonify({'error': 'Missing document or keyword'}), 400

    try:
        document = MELDocument.from_dict(data['document'])
        results = document.search_items(data['keyword'])

        return jsonify({
            'success': True,
            'results': [item.to_dict() for item in results],
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/<format_type>', methods=['POST'])
def export_document(format_type):
    """Export document to various formats."""
    data = request.get_json()

    if not data or 'document' not in data:
        return jsonify({'error': 'Missing document'}), 400

    try:
        document = MELDocument.from_dict(data['document'])

        if format_type == 'json':
            return jsonify({
                'success': True,
                'content': document.to_dict(),
                'filename': f'mel_{document.aircraft_type}_{datetime.now().strftime("%Y%m%d")}.json'
            })

        elif format_type == 'markdown':
            md_content = export_to_markdown(document)
            return jsonify({
                'success': True,
                'content': md_content,
                'filename': f'mel_{document.aircraft_type}_{datetime.now().strftime("%Y%m%d")}.md'
            })

        else:
            return jsonify({'error': f'Unknown format: {format_type}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    print(f"Starting MEL/MMEL Parser Web Interface on port {port}")
    print(f"Open http://localhost:{port} in your browser")

    app.run(host='0.0.0.0', port=port, debug=debug)
