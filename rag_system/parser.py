"""PDF parsing with Docling."""
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import shutil

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from rich.console import Console

from .config import IMAGE_DIR

console = Console()


@dataclass
class ParsedContent:
    """Container for parsed PDF content."""
    text: str
    tables: List[str]  # Markdown formatted tables
    images: List[Dict[str, Any]]  # {path: str, page: int, description: str}
    source_file: str


def parse_pdf(pdf_path: str | Path) -> ParsedContent:
    """
    Parse a PDF file using Docling.
    
    Extracts:
    - Text content
    - Tables (converted to Markdown)
    - Image locations
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        ParsedContent with extracted data
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    console.print(f"[blue]Parsing PDF:[/blue] {pdf_path.name}")
    
    # Configure pipeline for image extraction
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # Convert document
    result = converter.convert(pdf_path)
    doc = result.document
    
    # Extract text as markdown
    text_content = doc.export_to_markdown()
    
    # Extract tables separately (already in markdown from export)
    tables = []
    for table in doc.tables:
        table_md = table.export_to_markdown()
        if table_md:
            tables.append(table_md)
    
    # Extract images
    images = []
    doc_image_dir = IMAGE_DIR / pdf_path.stem
    doc_image_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, picture in enumerate(doc.pictures):
        # Save picture to disk
        image_filename = f"image_{idx + 1}.png"
        image_path = doc_image_dir / image_filename
        
        # Get the image if available
        if hasattr(picture, 'image') and picture.image is not None:
            picture.image.pil_image.save(str(image_path))
            images.append({
                "path": str(image_path),
                "index": idx + 1,
                "description": ""  # Will be filled by image describer
            })
            console.print(f"  [green]Extracted image:[/green] {image_filename}")
    
    console.print(f"  [green]✓[/green] Text extracted ({len(text_content)} chars)")
    console.print(f"  [green]✓[/green] {len(tables)} tables found")
    console.print(f"  [green]✓[/green] {len(images)} images extracted")
    
    # Save markdown to file
    markdown_dir = IMAGE_DIR.parent / "markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = markdown_dir / f"{pdf_path.stem}.md"
    markdown_path.write_text(text_content, encoding="utf-8")
    console.print(f"  [green]✓[/green] Markdown saved to: {markdown_path}")
    
    return ParsedContent(
        text=text_content,
        tables=tables,
        images=images,
        source_file=str(pdf_path)
    )

