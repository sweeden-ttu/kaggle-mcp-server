"""Top-level re-export of docling worker for backward compatibility."""

from mlsyseng_mcp.docling_worker import (
    discover_chapters,
    extract_all_chapters,
    extract_concepts,
    extract_pdf_content,
)

__all__ = [
    "discover_chapters",
    "extract_all_chapters",
    "extract_concepts",
    "extract_pdf_content",
]
