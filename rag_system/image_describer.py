"""Image description using Groq's Llama 3.2 Vision."""
import base64
from pathlib import Path
from typing import List, Dict, Any

from groq import Groq
from rich.console import Console

from .config import GROQ_API_KEY, VISION_MODEL, IMAGE_ANALYSIS_PROMPT

console = Console()

# Groq client
_client = None


def get_client() -> Groq:
    """Get or initialize the Groq client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Please set it in .env file or environment.")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def encode_image_base64(image_path: str | Path) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str | Path) -> str:
    """Get the media type for an image based on its extension."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return media_types.get(ext, "image/png")


def describe_image(image_path: str | Path) -> str:
    """
    Generate a text description of an image using Groq's Llama 3.2 Vision.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Text description of the image
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return f"[Image not found: {image_path}]"
    
    try:
        client = get_client()
        
        # Encode image to base64
        image_data = encode_image_base64(image_path)
        media_type = get_image_media_type(image_path)
        
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": IMAGE_ANALYSIS_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not describe image {image_path.name}: {e}[/yellow]")
        return f"[Image: {image_path.name}]"


def describe_all_images(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate descriptions for all images using Groq Vision API.
    
    Args:
        images: List of image dicts with 'path' key
        
    Returns:
        Updated list with 'description' filled in
    """
    if not images:
        return images
    
    console.print(f"[blue]Describing {len(images)} images with Groq Vision API...[/blue]")
    
    for i, img in enumerate(images):
        console.print(f"  Processing image {i + 1}/{len(images)}...")
        description = describe_image(img["path"])
        img["description"] = description
        console.print(f"    [green]✓[/green] Generated description ({len(description)} chars)")
    
    return images
