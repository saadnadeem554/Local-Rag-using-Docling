"""Main RAG pipeline orchestration."""
from pathlib import Path
from typing import Dict, Any
from rich.console import Console

from .config import TOP_K_RETRIEVAL, TOP_K_RERANK, IMAGE_DIR
from .parser import parse_pdf
from .image_describer import describe_all_images
from .chunker import create_chunks
from .embedder import embed_query
from .vector_store import add_chunks, search, clear_collection, get_stats
from .reranker import rerank
from .generator import generate_response, build_context, stream_response

console = Console()


def save_image_descriptions_to_markdown(pdf_path: Path, images: list) -> None:
    """Append AI-generated image descriptions to the markdown file."""
    if not images:
        return
    
    markdown_dir = IMAGE_DIR.parent / "markdown"
    markdown_path = markdown_dir / f"{pdf_path.stem}.md"
    
    if not markdown_path.exists():
        return
    
    # Build image descriptions section
    descriptions = ["\n\n---\n\n## AI-Generated Image Descriptions\n"]
    for img in images:
        img_path = img.get("path", "unknown")
        description = img.get("description", "No description available")
        descriptions.append(f"\n### Image: {Path(img_path).name}\n")
        descriptions.append(f"**Path:** `{img_path}`\n\n")
        descriptions.append(f"{description}\n")
    
    # Append to markdown file
    with open(markdown_path, "a", encoding="utf-8") as f:
        f.writelines(descriptions)
    
    console.print(f"  [green]✓[/green] Image descriptions added to markdown")


def ingest_pdf(pdf_path: str | Path) -> Dict[str, Any]:
    """
    Ingest a PDF into the RAG system.
    
    Steps:
    1. Parse PDF with Docling
    2. Describe images with Groq Vision
    3. Chunk content
    4. Embed and store in ChromaDB
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Ingestion statistics
    """
    pdf_path = Path(pdf_path)
    console.print(f"\n[bold blue]═══ Ingesting PDF: {pdf_path.name} ═══[/bold blue]\n")
    
    # Step 1: Parse PDF
    console.print("[bold]Step 1/4: Parsing PDF[/bold]")
    parsed = parse_pdf(pdf_path)
    
    # Step 2: Describe images
    console.print("\n[bold]Step 2/4: Describing Images[/bold]")
    if parsed.images:
        parsed.images = describe_all_images(parsed.images)
        # Save image descriptions to markdown
        save_image_descriptions_to_markdown(pdf_path, parsed.images)
    else:
        console.print("  No images to describe")
    
    # Step 3: Create chunks
    console.print("\n[bold]Step 3/4: Creating Chunks[/bold]")
    chunks = create_chunks(parsed)
    console.print(f"  Created {len(chunks)} chunks")
    
    # Step 4: Store in vector DB
    console.print("\n[bold]Step 4/4: Storing in Vector Database[/bold]")
    num_added = add_chunks(chunks)
    
    stats = get_stats()
    console.print(f"\n[bold green]✓ Ingestion complete![/bold green]")
    console.print(f"  Total documents in store: {stats['total_documents']}")
    
    return {
        "file": str(pdf_path),
        "text_length": len(parsed.text),
        "tables": len(parsed.tables),
        "images": len(parsed.images),
        "chunks_created": len(chunks),
        "chunks_added": num_added,
        "total_in_store": stats["total_documents"]
    }


def query(question: str, stream: bool = False) -> str | None:
    """
    Query the RAG system.
    
    Steps:
    1. Embed query
    2. Retrieve Top-K from ChromaDB
    3. Rerank with FlashRank
    4. Generate response with Llama-3.2
    
    Args:
        question: User's question
        stream: If True, stream the response
        
    Returns:
        Generated response (or None if streaming)
    """
    console.print(f"\n[bold blue]═══ Processing Query ═══[/bold blue]\n")
    console.print(f"[italic]{question}[/italic]\n")
    
    # Step 1: Embed query
    console.print("[bold]Step 1/4: Embedding Query[/bold]")
    query_embedding = embed_query(question)
    console.print("  [green]✓[/green] Query embedded")
    
    # Step 2: Retrieve from vector store
    console.print(f"\n[bold]Step 2/4: Retrieving Top-{TOP_K_RETRIEVAL} Results[/bold]")
    results = search(query_embedding, top_k=TOP_K_RETRIEVAL)
    console.print(f"  [green]✓[/green] Retrieved {len(results)} results")
    
    if not results:
        console.print("[yellow]No relevant documents found. Try ingesting some PDFs first.[/yellow]")
        return "No relevant documents found in the knowledge base."
    
    # Show top 5 BEFORE reranking (ChromaDB similarity)
    console.print("\n[bold cyan]Before Reranking (ChromaDB Similarity - lower distance = better):[/bold cyan]")
    for i, r in enumerate(results[:5], 1):
        chunk_type = r.get("metadata", {}).get("chunk_type", "text")
        distance = r.get("distance", 0)
        preview = r["content"][:80].replace("\n", " ")
        console.print(f"  {i}. [{chunk_type}] (dist: {distance:.4f}) {preview}...")
    
    # Step 3: Rerank
    console.print(f"\n[bold]Step 3/4: Reranking to Top-{TOP_K_RERANK}[/bold]")
    top_results = rerank(question, results, top_k=TOP_K_RERANK)
    console.print(f"  [green]✓[/green] Reranked to {len(top_results)} results")
    
    # Show AFTER reranking (FlashRank scores - higher = better)
    console.print("\n[bold green]After Reranking (FlashRank Score - higher = better):[/bold green]")
    for i, r in enumerate(top_results, 1):
        chunk_type = r.get("metadata", {}).get("chunk_type", "text")
        score = r.get("rerank_score", 0)
        preview = r["content"][:80].replace("\n", " ")
        console.print(f"  {i}. [{chunk_type}] (score: {score:.4f}) {preview}...")
    
    # Step 4: Generate response
    console.print(f"\n[bold]Step 4/4: Generating Response[/bold]")
    context = build_context(top_results)
    
    if stream:
        console.print("\n[bold green]Response:[/bold green]")
        full_response = ""
        for chunk in stream_response(question, context):
            console.print(chunk, end="")
            full_response += chunk
        console.print("\n")
        return full_response
    else:
        response = generate_response(question, context)
        console.print(f"\n[bold green]Response:[/bold green]\n{response}\n")
        return response


def reset():
    """Clear all data from the vector store."""
    console.print("[yellow]Clearing vector store...[/yellow]")
    clear_collection()
    console.print("[green]✓ Vector store reset[/green]")
