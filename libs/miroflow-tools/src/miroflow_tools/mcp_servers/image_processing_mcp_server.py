# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Image Processing MCP Server for MiroThinker

This server provides basic image processing tools for multi-modal search framework,
inspired by DeepEyesV2's approach to image manipulation.

Tools provided:
- ZoomIn: Enlarge a specific region of an image
- Rotation: Rotate an image by a specified angle
- Flip: Flip an image horizontally or vertically
- PutBox: Add a bounding box annotation to an image

IMPORTANT: All image processing tools can ONLY process images that have already been
loaded into the agent's context (through search results, fetch_image, or other image tools).
Do NOT use these tools on images that the agent has not yet viewed.
"""

import base64
import io
import logging
import os
import random
import string
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Ensure .env file is loaded
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("image-processing-server")

# Constants
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
MIN_IMAGE_SIDE = 28
MAX_IMAGE_SIDE = 2048


class OSSUploader:
    """Handler for uploading images to Aliyun OSS."""

    def __init__(self):
        """Initialize OSS uploader with credentials from environment variables."""
        self.access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        self.access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        self.bucket_name = os.getenv("OSS_BUCKET_NAME", "")
        self.endpoint = os.getenv("OSS_ENDPOINT", "")

        # Import oss2 only when needed to avoid dependency issues
        try:
            import oss2

            self.oss2 = oss2
        except ImportError:
            print("Warning: oss2 not installed. OSS upload will be disabled.")
            self.oss2 = None

    def generate_random_string(self, length=32):
        """
        Generate a random string for file naming.

        Args:
            length: Length of the random string

        Returns:
            Random alphanumeric string
        """
        letters = string.ascii_letters + string.digits
        return "".join(random.choice(letters) for _ in range(length))

    def upload_bytes(self, image_bytes: bytes) -> Optional[str]:
        """
        Upload image bytes to Aliyun OSS.

        Args:
            image_bytes: Image data in bytes format

        Returns:
            Signed OSS URL (valid for 100 hours), or None if upload fails
        """
        if not self.oss2:
            print("Error: oss2 not available. Cannot upload to OSS.")
            return None

        if not self.access_key_id or not self.access_key_secret:
            print("Error: OSS credentials not configured.")
            return None

        try:
            image_name = f"{self.generate_random_string()}.jpeg"
            target_path = f"/image/{image_name}"

            # Check file size
            image_size = len(image_bytes)

            # Skip small files (< 1KB)
            if image_size <= 1024:
                print("Info: Image size too small (< 1KB), skipping upload.")
                return None

            # Authenticate and create bucket
            auth = self.oss2.Auth(self.access_key_id, self.access_key_secret)
            bucket = self.oss2.Bucket(auth, self.endpoint, self.bucket_name)

            # Upload
            bucket.put_object(target_path, image_bytes)

            # Generate signed URL (valid for 100 hours)
            file_url = bucket.sign_url("GET", target_path, 360000)
            return file_url

        except Exception as e:
            print(f"Error: Failed to upload image to OSS: {str(e)}")
            return None


def format_image_for_context(
    image_base64: str,
    image_url: Optional[str],
    description: str,
) -> Tuple[dict, str]:
    """
    Format image data for inclusion in LLM context.

    Args:
        image_base64: Base64-encoded image data
        image_url: URL of the image (can be None)
        description: Text description of the image

    Returns:
        Tuple of (image_content_dict, text_description)
    """
    # Create image content dict for OpenAI API format
    image_content = {
        "type": "image_url",
        "image_url": {"url": image_base64},
    }

    # Create text description with metadata
    text_parts = [f"Image URL: {image_url or 'N/A'}", f"Description: {description}"]

    text_description = ", ".join(text_parts)

    return image_content, text_description


def validate_image_content(image_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate that the downloaded content is actually an image.
    
    Args:
        image_bytes: The raw bytes of the downloaded content
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not image_bytes or len(image_bytes) < 8:
        return False, "Downloaded content is empty or too small to be a valid image"
    
    # Check for HTML content (common when server returns error page)
    content_start = image_bytes[:100].strip().lower()
    if content_start.startswith((b'<!doctype', b'<html', b'<head', b'<?xml')):
        return False, "Downloaded content is HTML/XML, not an image. The URL may have returned an error page or redirect."
    
    return True, ""


def download_image_from_url(image_url: str, timeout: int = 30) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Download image from URL and return bytes and error message.

    Args:
        image_url: URL of the image to download
        timeout: Request timeout in seconds

    Returns:
        Tuple of (image_bytes, error_message)
        - If successful: (bytes, None)
        - If failed: (None, error_message)
    """
    try:
        # Validate URL format
        parsed_url = urlparse(image_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return None, f"Invalid URL format: {image_url}"

        if parsed_url.scheme not in ['http', 'https']:
            return None, f"Unsupported URL scheme: {parsed_url.scheme}. Only http and https are supported."

        # Download image with User-Agent header to avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(image_url, timeout=timeout, stream=True, headers=headers)
        response.raise_for_status()

        # Check Content-Type header - reject if it's HTML/text
        content_type = response.headers.get('Content-Type', '')
        if content_type.startswith('text/html') or content_type.startswith('text/plain'):
            return None, f"[ERROR]: URL returned {content_type} instead of an image. The server may have returned an error page or redirect."

        # Check file size
        file_size = len(response.content)
        if file_size > MAX_IMAGE_SIZE:
            return None, f"[ERROR]: Downloaded image size ({file_size / (1024 * 1024):.2f}MB) exceeds maximum allowed size (20MB)"

        # Validate the content is actually an image
        is_valid, validation_error = validate_image_content(response.content)
        if not is_valid:
            return None, f"[ERROR]: {validation_error}"

        return response.content, None

    except requests.exceptions.Timeout:
        return None, f"[ERROR]: Download timeout after {timeout} seconds"
    except requests.exceptions.RequestException as e:
        return None, f"[ERROR]: Download failed: {str(e)}"
    except Exception as e:
        return None, f"[ERROR]: Unexpected error: {str(e)}"


def _ensure_pil_image_dimensions(img: Image.Image) -> Image.Image:
    """Resize a PIL Image so shortest side >= 28 and longest side <= 2048."""
    w, h = img.size
    if min(w, h) >= MIN_IMAGE_SIDE and max(w, h) <= MAX_IMAGE_SIDE:
        return img
    scale = 1.0
    if max(w, h) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(w, h)
    if min(w * scale, h * scale) < MIN_IMAGE_SIDE:
        scale = MIN_IMAGE_SIDE / min(w, h)
    new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
    if new_w == w and new_h == h:
        return img
    logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
    return img.resize((new_w, new_h), Image.LANCZOS)


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode a PIL Image to base64 string.
    Automatically resizes so shortest side >= 28 and longest side <= 2048.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Base64-encoded image string with data URI prefix
    """
    image = _ensure_pil_image_dimensions(image)

    buffer = io.BytesIO()
    image.save(buffer, format=format)
    image_bytes = buffer.getvalue()
    base64_str = base64.b64encode(image_bytes).decode("utf-8")

    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{base64_str}"


def load_image_from_url(image_url: str) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Load an image from URL.

    Args:
        image_url: URL of the image to load

    Returns:
        Tuple of (PIL Image, error_message). Image is None if error occurs.
    """
    try:
        # Download image from URL
        image_bytes, error = download_image_from_url(image_url)
        if error:
            return None, error

        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        return image, None

    except Exception as e:
        return None, f"[ERROR]: Failed to load image from URL: {str(e)}"


@mcp.tool()
async def zoom_in(
    image_url: str,
    x: int,
    y: int,
    width: int,
    height: int,
    output_format: str = "PNG",
) -> str:
    """Zoom in on a specific rectangular region of an image.

    Args:
        image_url: URL of the image to process (must start with http:// or https://)
        x: X coordinate of the top-left corner (in pixels)
        y: Y coordinate of the top-left corner (in pixels)
        width: Width of the region (in pixels)
        height: Height of the region (in pixels)
        output_format: Output format ('PNG', 'JPEG', etc.). Default: 'PNG'

    Returns:
        JSON-formatted multi-modal content with text description and base64-encoded image
    """
    # Load image from URL
    image, error = load_image_from_url(image_url)
    if error:
        return error

    try:
        # Validate coordinates
        img_width, img_height = image.size
        if x < 0 or y < 0:
            return f'[{{"error": "Coordinates cannot be negative (x={x}, y={y})"}}]'
        if width <= 0 or height <= 0:
            return f'[{{"error": "Width and height must be positive (width={width}, height={height})"}}]'
        if x + width > img_width or y + height > img_height:
            return f'[{{"error": "Your cropped region extends beyond image bounds (image size: {img_width}x{img_height}, your cropped region: ({x}+{width})x({y}+{height}))"}}]'

        MIN_DIMENSION = 28
        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            return (
                f'[{{"error": "Cropped region is too small: {width}x{height} pixels. '
                f'Each dimension must be at least {MIN_DIMENSION} pixels. '
                f'Please select a larger region."}}]'
            )

        # Crop the region
        box = (x, y, x + width, y + height)
        cropped = image.crop(box)

        # Convert to bytes for OSS upload
        buffer = io.BytesIO()
        cropped.save(buffer, format=output_format)
        image_bytes = buffer.getvalue()

        # Upload to OSS
        oss_uploader = OSSUploader()
        oss_url = oss_uploader.upload_bytes(image_bytes)

        # Encode to base64 for return
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = f"image/{output_format.lower()}"
        base64_with_mime = f"data:{mime_type};base64,{base64_str}"

        # Generate description
        description = f"[Zoomed region: ({x}, {y}, {width}, {height}) from original image]"

        # Format for multi-modal context
        image_content, text_description = format_image_for_context(
            base64_with_mime, oss_url, description
        )

        # Return multi-modal format as JSON string (similar to fetch_image)
        import json
        result = [
            {
                "type": "text",
                "text": f"[SUCCESS]: Zoomed in on region ({x}, {y}, {width}, {height}). Image dimensions: {width}x{height}.\n{text_description}"
            },
            image_content
        ]

        return json.dumps(result)

    except Exception as e:
        return f'[{{"error": "Failed to zoom in: {str(e)}"}}]'


@mcp.tool()
async def rotate(
    image_url: str,
    angle: float,
    expand: bool = False,
    output_format: str = "PNG",
) -> str:
    """Rotate an image by a specified angle.

    Args:
        image_url: URL of the image to process (must start with http:// or https://)
        angle: Rotation angle in degrees (positive: counter-clockwise, negative: clockwise)
        expand: If True, expands output to fit entire rotated image. Default: False
        output_format: Output format ('PNG', 'JPEG', etc.). Default: 'PNG'

    Returns:
        JSON-formatted multi-modal content with text description and base64-encoded image
    """
    # Load image from URL
    image, error = load_image_from_url(image_url)
    if error:
        return error

    try:
        # Rotate image
        rotated = image.rotate(angle, expand=expand, resample=Image.BICUBIC)

        # Convert to bytes for OSS upload
        buffer = io.BytesIO()
        rotated.save(buffer, format=output_format)
        image_bytes = buffer.getvalue()

        # Upload to OSS
        oss_uploader = OSSUploader()
        oss_url = oss_uploader.upload_bytes(image_bytes)

        # Encode to base64 for return
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = f"image/{output_format.lower()}"
        base64_with_mime = f"data:{mime_type};base64,{base64_str}"

        # Generate description
        expand_info = "expanded" if expand else "original size"
        description = f"[Rotated image: {angle} degrees, {expand_info}]"

        # Format for multi-modal context
        image_content, text_description = format_image_for_context(
            base64_with_mime, oss_url, description
        )

        original_size = image.size
        rotated_size = rotated.size

        # Return multi-modal format as JSON string (similar to fetch_image)
        import json
        result = [
            {
                "type": "text",
                "text": f"[SUCCESS]: Rotated image by {angle} degrees ({'expanded' if expand else 'original size'}). Original size: {original_size[0]}x{original_size[1]}, Rotated size: {rotated_size[0]}x{rotated_size[1]}.\n{text_description}"
            },
            image_content
        ]

        return json.dumps(result)

    except Exception as e:
        return f'[{{"error": "Failed to rotate: {str(e)}"}}]'


@mcp.tool()
async def flip(
    image_url: str,
    direction: str,
    output_format: str = "PNG",
) -> str:
    """Flip an image horizontally or vertically.

    Args:
        image_url: URL of the image to process (must start with http:// or https://)
        direction: 'horizontal' for left-right flip, 'vertical' for top-bottom flip
        output_format: Output format ('PNG', 'JPEG', etc.). Default: 'PNG'

    Returns:
        JSON-formatted multi-modal content with text description and base64-encoded image
    """
    # Load image from URL
    image, error = load_image_from_url(image_url)
    if error:
        return error

    try:
        # Validate direction
        direction = direction.lower()
        if direction not in ["horizontal", "vertical"]:
            return f'[{{"error": "Invalid direction \'{direction}\'. Must be \'horizontal\' or \'vertical\'"}}]'

        # Flip image
        if direction == "horizontal":
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            direction_str = "horizontally (left-right)"
        else:  # vertical
            flipped = image.transpose(Image.FLIP_TOP_BOTTOM)
            direction_str = "vertically (top-bottom)"

        # Convert to bytes for OSS upload
        buffer = io.BytesIO()
        flipped.save(buffer, format=output_format)
        image_bytes = buffer.getvalue()

        # Upload to OSS
        oss_uploader = OSSUploader()
        oss_url = oss_uploader.upload_bytes(image_bytes)

        # Encode to base64 for return
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = f"image/{output_format.lower()}"
        base64_with_mime = f"data:{mime_type};base64,{base64_str}"

        # Generate description
        description = f"[Flipped image: {direction_str}]"

        # Format for multi-modal context
        image_content, text_description = format_image_for_context(
            base64_with_mime, oss_url, description
        )

        # Return multi-modal format as JSON string (similar to fetch_image)
        import json
        result = [
            {
                "type": "text",
                "text": f"[SUCCESS]: Flipped image {direction_str}. Image dimensions: {flipped.size[0]}x{flipped.size[1]}.\n{text_description}"
            },
            image_content
        ]

        return json.dumps(result)

    except Exception as e:
        return f'[{{"error": "Failed to flip: {str(e)}"}}]'


@mcp.tool()
async def put_box(
    image_url: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "red",
    line_width: int = 3,
    label: Optional[str] = None,
    output_format: str = "PNG",
) -> str:
    """Add a bounding box annotation to an image.

    Args:
        image_url: URL of the image to process (must start with http:// or https://)
        x1: X coordinate of the top-left corner (in pixels)
        y1: Y coordinate of the top-left corner (in pixels)
        x2: X coordinate of the bottom-right corner (in pixels)
        y2: Y coordinate of the bottom-right corner (in pixels)
        color: Box color (name or hex code). Default: 'red'
        line_width: Box border width in pixels. Default: 3
        label: Optional text label above the box
        output_format: Output format ('PNG', 'JPEG', etc.). Default: 'PNG'

    Returns:
        JSON-formatted multi-modal content with text description and base64-encoded image
    """
    # Load image from URL
    image, error = load_image_from_url(image_url)
    if error:
        return error

    try:
        # Validate coordinates
        img_width, img_height = image.size
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            return f'[{{"error": "Coordinates cannot be negative (x1={x1}, y1={y1}, x2={x2}, y2={y2})"}}]'

        if x1 >= x2 or y1 >= y2:
            return f'[{{"error": "Invalid box dimensions: x1 must be less than x2, y1 must be less than y2 (got: x1={x1}, y1={y1}, x2={x2}, y2={y2})"}}]'

        if x2 > img_width or y2 > img_height:
            return f'[{{"error": "Box extends beyond image bounds (image size: {img_width}x{img_height}, box: ({x1},{y1}) to ({x2},{y2}))"}}]'

        # Create a copy to avoid modifying the original
        annotated = image.copy()

        # Draw the box
        draw = ImageDraw.Draw(annotated)
        box_coords = [(x1, y1), (x2, y2)]
        draw.rectangle(box_coords, outline=color, width=line_width)

        # Add label if provided
        if label:
            # Get text size (this is approximate, works for most cases)
            try:
                # Try to use a default font
                from PIL import ImageFont

                font = ImageFont.load_default()
                text_bbox = draw.textbbox((x1, y1), label, font=font)
                text_height = text_bbox[3] - text_bbox[1]

                # Draw text background
                text_position = (x1, max(0, y1 - text_height - 4))
                draw.text(text_position, label, fill=color, font=font)
            except Exception:
                # If font loading fails, just draw the text without font
                draw.text((x1, max(0, y1 - 20)), label, fill=color)

        # Convert to bytes for OSS upload
        buffer = io.BytesIO()
        annotated.save(buffer, format=output_format)
        image_bytes = buffer.getvalue()

        # Upload to OSS
        oss_uploader = OSSUploader()
        oss_url = oss_uploader.upload_bytes(image_bytes)

        # Encode to base64 for return
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = f"image/{output_format.lower()}"
        base64_with_mime = f"data:{mime_type};base64,{base64_str}"

        # Generate description
        label_info = f" with label '{label}'" if label else ""
        description = f"[Annotated image: bounding box ({x1},{y1}) to ({x2},{y2}){label_info}, color: {color}]"

        # Format for multi-modal context
        image_content, text_description = format_image_for_context(
            base64_with_mime, oss_url, description
        )

        # Return multi-modal format as JSON string (similar to fetch_image)
        import json
        result = [
            {
                "type": "text",
                "text": f"[SUCCESS]: Added bounding box from ({x1},{y1}) to ({x2},{y2}){label_info}. Box color: {color}, Line width: {line_width}.\n{text_description}"
            },
            image_content
        ]

        return json.dumps(result)

    except Exception as e:
        return f'[{{"error": "Failed to put box: {str(e)}"}}]'


@mcp.tool()
async def get_image_info(image_url: str) -> str:
    """Get basic information about an image.

    Args:
        image_url: URL of the image to analyze (must start with http:// or https://)

    Returns:
        String with image information (dimensions, format, mode)
    """
    # Load image from URL
    image, error = load_image_from_url(image_url)
    if error:
        return error

    try:
        width, height = image.size
        image_format = image.format or "Unknown"
        mode = image.mode

        info = f"""[SUCCESS]: Image Information
Dimensions: {width} x {height} pixels
Format: {image_format}
Mode: {mode}"""

        return info

    except Exception as e:
        return f"[ERROR]: Failed to get image info: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
