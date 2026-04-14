# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Image processing utilities for multi-modal support.

This module provides functions for:
- Uploading images to Aliyun OSS (with local HTTP server fallback)
- Downloading and encoding images
- Generating image descriptions for multi-modal context
"""

import base64
import io
import logging
import os
import random
import shutil
import socket
import string
import tempfile
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from io import BytesIO
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv
from miroflow_tools.image_llm_payload import ensure_image_base64_under_limit
from PIL import Image

logger = logging.getLogger(__name__)

# Ensure .env file is loaded
load_dotenv()

MIN_IMAGE_SIDE = 28
MAX_IMAGE_SIDE = 2048


def ensure_image_dimensions(
    image_bytes: bytes,
    min_side: int = MIN_IMAGE_SIDE,
    max_side: int = MAX_IMAGE_SIDE,
) -> bytes:
    """Resize image bytes so that the shortest side >= *min_side* and the
    longest side <= *max_side*.  Returns the (possibly modified) image as
    PNG bytes.  If the input is already within bounds, it is returned
    unchanged.

    The function preserves aspect ratio.  When both constraints cannot be
    satisfied simultaneously (image would need to be both up- and down-scaled),
    the *max_side* constraint takes priority.
    """
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    short_side = min(w, h)
    long_side = max(w, h)

    if short_side >= min_side and long_side <= max_side:
        return image_bytes

    scale = 1.0
    if long_side > max_side:
        scale = max_side / long_side
    if min(w * scale, h * scale) < min_side:
        scale = min_side / min(w, h)

    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)

    if new_w == w and new_h == h:
        return image_bytes

    logger.info(
        f"Resizing image from {w}x{h} to {new_w}x{new_h} "
        f"(min_side={min_side}, max_side={max_side})"
    )
    img = img.resize((new_w, new_h), Image.LANCZOS)

    out = io.BytesIO()
    fmt = img.format or "PNG"
    if img.mode == "RGBA" and fmt.upper() == "JPEG":
        img = img.convert("RGB")
    img.save(out, format=fmt)
    return out.getvalue()


class _SilentHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler that suppresses access log output."""

    def log_message(self, format, *args):
        pass


class LocalImageServer:
    """
    Singleton HTTP server for serving local images when OSS is unavailable.
    Starts a background thread HTTP server and copies files to a serve directory.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.serve_dir = tempfile.mkdtemp(prefix="miroflow_images_")
        self.port = self._find_free_port()
        self._start_server()

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _start_server(self):
        handler = partial(_SilentHTTPHandler, directory=self.serve_dir)
        self.server = HTTPServer(("0.0.0.0", self.port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"Info: Local image server started on port {self.port}")

    def serve_file(self, file_path: str) -> str:
        """Copy a file to the serve directory and return its localhost URL."""
        filename = os.path.basename(file_path)
        random_prefix = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=8)
        )
        served_name = f"{random_prefix}_{filename}"
        dest = os.path.join(self.serve_dir, served_name)
        shutil.copy2(file_path, dest)
        return f"http://localhost:{self.port}/{served_name}"

    def serve_bytes(self, data: bytes, suffix: str = ".jpeg") -> str:
        """Write bytes to the serve directory and return its localhost URL."""
        random_prefix = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=8)
        )
        served_name = f"{random_prefix}{suffix}"
        dest = os.path.join(self.serve_dir, served_name)
        with open(dest, "wb") as f:
            f.write(data)
        return f"http://localhost:{self.port}/{served_name}"


class OSSUploader:
    """Handler for uploading images to Aliyun OSS, with local HTTP server fallback."""

    _oss_failed = False

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

    def _upload_to_oss(self, image, byte=False) -> Optional[str]:
        """Attempt to upload to Aliyun OSS. Returns URL or None on failure."""
        if OSSUploader._oss_failed:
            return None

        if not self.oss2:
            return None

        if not self.access_key_id or not self.access_key_secret:
            return None

        try:
            image_name = f"{self.generate_random_string()}.jpeg"
            target_path = f"image/{image_name}"

            if byte:
                image_bytes = BytesIO(image)
                image_size = image_bytes.getbuffer().nbytes
            else:
                image_size = os.path.getsize(image)

            if image_size <= 1024:
                print("Info: Image size too small (< 1KB), skipping upload.")
                return None

            auth = self.oss2.Auth(self.access_key_id, self.access_key_secret)
            bucket = self.oss2.Bucket(auth, self.endpoint, self.bucket_name)

            if byte:
                bucket.put_object(target_path, image_bytes.getvalue())
            else:
                bucket.put_object_from_file(target_path, image)

            file_url = bucket.sign_url("GET", target_path, 360000)
            return file_url

        except Exception as e:
            print(f"Warning: OSS upload failed: {str(e)}")
            OSSUploader._oss_failed = True
            print("Info: OSS marked as unavailable; subsequent uploads will use local server directly.")
            return None

    def _upload_via_local_server(self, image, byte=False) -> Optional[str]:
        """Fallback: serve image via a local HTTP server."""
        try:
            server = LocalImageServer()
            if byte:
                return server.serve_bytes(image)
            else:
                return server.serve_file(image)
        except Exception as e:
            print(f"Error: Local image server fallback also failed: {str(e)}")
            return None

    def upload(self, image, byte=False) -> Optional[str]:
        """
        Upload image to get an HTTP URL. Tries OSS first, then falls back
        to a local HTTP server.

        Args:
            image: Image data (bytes or file path)
            byte: Whether the input is in byte format

        Returns:
            Image URL (OSS signed URL or localhost URL), or None if all methods fail
        """
        url = self._upload_to_oss(image, byte=byte)
        if url:
            return url

        print("Info: Falling back to local image server...")
        return self._upload_via_local_server(image, byte=byte)


def encode_image_to_base64(
    image_path: str,
) -> Optional[Tuple[str, str]]:
    """
    Encode an image file to base64 after dimension clamp and optional JPEG shrink.

    Args:
        image_path: Path to the image file

    Returns:
        (base64_payload, mime_type) or None if encoding fails. *mime_type* reflects
        re-encoding (e.g. image/jpeg) when the payload was clamped.
    """
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image file does not exist: {image_path}")
            return None

        file_size = os.path.getsize(image_path)
        if file_size == 0:
            print(f"Error: Image file is empty (0 bytes): {image_path}")
            return None

        with open(image_path, "rb") as image_file:
            raw_bytes = image_file.read()

        if len(raw_bytes) == 0:
            print(f"Error: Read 0 bytes from image file: {image_path}")
            return None

        raw_bytes = ensure_image_dimensions(raw_bytes)
        mime_hint = get_image_mime_type(image_path)
        raw_bytes, out_mime = ensure_image_base64_under_limit(
            raw_bytes, mime_type=mime_hint
        )

        encoded = base64.b64encode(raw_bytes).decode("utf-8")
        if not encoded:
            print(f"Error: base64 encoding produced empty result for: {image_path} (file_size={file_size})")
            return None

        print(f"Info: Encoded image {os.path.basename(image_path)}: file_size={file_size}, base64_len={len(encoded)}")
        return encoded, out_mime
    except Exception as e:
        print(f"Error: Failed to encode image to base64: {image_path}, error={e}")
        return None


def get_image_mime_type(image_path: str) -> str:
    """
    Get MIME type for an image file based on its extension.

    Args:
        image_path: Path to the image file

    Returns:
        MIME type string
    """
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")


def download_image_from_url(image_url: str, timeout=10) -> Optional[bytes]:
    """
    Download an image from URL.

    Args:
        image_url: URL of the image
        timeout: Request timeout in seconds

    Returns:
        Image bytes, or None if download fails
    """
    try:
        response = requests.get(image_url, timeout=timeout, stream=True)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error: Failed to download image from {image_url}: {str(e)}")
        return None


def encode_image_bytes_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Base64-encoded string
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def generate_simple_image_caption(image_path: str, task_description: str = "") -> str:
    """
    Generate a simple caption for an image (brief description only).

    Unlike the detailed caption generation, this creates a minimal caption
    for multi-modal context.

    Args:
        image_path: Path to the image file
        task_description: Optional task context

    Returns:
        Simple caption string
    """
    # For initial input images, we provide a minimal description
    # The actual visual understanding will be done by the multi-modal model
    return f"[Original input image: {os.path.basename(image_path)}]"


def format_image_for_context(
    image_base64: str,
    image_url: Optional[str],
    description: str,
    webpage_url: Optional[str] = None,
) -> Tuple[dict, str]:
    """
    Format image data for inclusion in LLM context.

    Args:
        image_base64: Base64-encoded image data
        image_url: URL of the image (can be None)
        description: Text description of the image
        webpage_url: URL of the webpage containing the image (optional)

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

    if webpage_url:
        text_parts.append(f"Webpage URL: {webpage_url}")

    text_description = ", ".join(text_parts)

    return image_content, text_description
