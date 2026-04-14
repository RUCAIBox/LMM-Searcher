# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
adapted from
https://github.com/MiroMindAI/MiroRL/blob/5073693549ffe05a157a1886e87650ef3be6606e/mirorl/tools/serper_search.py#L1
"""

import base64
import json
import logging
import os
from typing import Any, Dict, List

import requests
from mcp.server.fastmcp import FastMCP
from miroflow_tools.image_llm_payload import ensure_image_base64_under_limit
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .utils import decode_http_urls_in_dict

logger = logging.getLogger(__name__)


_LLM_SUPPORTED_MIME_TYPES = {
    'image/png', 'image/jpeg', 'image/gif', 'image/webp',
}


def _detect_image_mime(data: bytes) -> str | None:
    """Return mime type if data starts with a known LLM-supported image signature, else None."""
    if not data or len(data) < 8:
        return None
    sigs = [
        (b'\xFF\xD8\xFF', 'image/jpeg'),
        (b'\x89PNG\r\n\x1a\n', 'image/png'),
        (b'GIF87a', 'image/gif'),
        (b'GIF89a', 'image/gif'),
        (b'BM', 'image/bmp'),
        (b'RIFF', 'image/webp'),
    ]
    for sig, mime in sigs:
        if data.startswith(sig):
            if mime not in _LLM_SUPPORTED_MIME_TYPES:
                logger.warning(f"Skipping unsupported image format: {mime}")
                return None
            return mime
    content_start = data[:100].strip().lower()
    if content_start.startswith((b'<!doctype', b'<html', b'<head', b'<?xml')):
        return None
    try:
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(data))
        img.verify()
        mime = f"image/{img.format.lower()}" if img.format else "image/jpeg"
        if mime not in _LLM_SUPPORTED_MIME_TYPES:
            logger.warning(f"Skipping unsupported image format detected by PIL: {mime}")
            return None
        return mime
    except Exception:
        return None


_MIN_IMAGE_SIDE = 28
_MAX_IMAGE_SIDE = 2048


def _ensure_image_dimensions(image_bytes: bytes) -> bytes:
    """Resize so shortest side >= 28 and longest side <= 2048. Returns PNG bytes."""
    try:
        from PIL import Image as _PILImage
        from io import BytesIO as _BytesIO
        img = _PILImage.open(_BytesIO(image_bytes))
        w, h = img.size
        short_side, long_side = min(w, h), max(w, h)
        if short_side >= _MIN_IMAGE_SIDE and long_side <= _MAX_IMAGE_SIDE:
            return image_bytes
        scale = 1.0
        if long_side > _MAX_IMAGE_SIDE:
            scale = _MAX_IMAGE_SIDE / long_side
        if min(w * scale, h * scale) < _MIN_IMAGE_SIDE:
            scale = _MIN_IMAGE_SIDE / min(w, h)
        new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
        if new_w == w and new_h == h:
            return image_bytes
        logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
        img = img.resize((new_w, new_h), _PILImage.LANCZOS)
        out = _BytesIO()
        fmt = img.format or "PNG"
        if img.mode == "RGBA" and fmt.upper() == "JPEG":
            img = img.convert("RGB")
        img.save(out, format=fmt)
        return out.getvalue()
    except Exception as e:
        logger.warning(f"Failed to resize image: {e}")
        return image_bytes


def download_and_encode_images(
    image_results: List[Dict[str, Any]], max_images: int = 5, limit_results: bool = True
) -> List[Dict[str, Any]]:
    """
    Download and encode images to base64 format.

    Args:
        image_results: List of image search results with 'imageUrl' field
        max_images: Maximum number of images to process (default: 5)
        limit_results: If True, only return max_images results; if False, return all results but only encode first max_images (default: True)

    Returns:
        List of image results with added base64 data
    """
    processed_results = []

    for idx, result in enumerate(image_results[:max_images]):
        image_url = result.get("imageUrl") or result.get("link", "")
        if not image_url:
            continue

        try:
            response = requests.get(image_url, timeout=10, stream=True)
            response.raise_for_status()

            raw_bytes = response.content
            detected_mime = _detect_image_mime(raw_bytes)
            if detected_mime is None:
                content_type = response.headers.get("content-type", "")
                logger.warning(
                    f"Skipping invalid image {idx + 1}: url={image_url}, "
                    f"content_type={content_type}, size={len(raw_bytes)}, "
                    f"first_20_bytes={raw_bytes[:20]!r}"
                )
                processed_results.append(result)
                continue

            raw_bytes = _ensure_image_dimensions(raw_bytes)
            raw_bytes, out_mime = ensure_image_base64_under_limit(
                raw_bytes, mime_type=detected_mime
            )
            image_base64 = base64.b64encode(raw_bytes).decode("utf-8")
            image_base64_with_mime = f"data:{out_mime};base64,{image_base64}"

            result_copy = result.copy()
            result_copy["base64_data"] = image_base64_with_mime
            processed_results.append(result_copy)

        except Exception as e:
            logger.warning(f"Failed to download/encode image {idx + 1} ({image_url}): {e}")
            processed_results.append(result)

    if not limit_results and len(image_results) > max_images:
        processed_results.extend(image_results[max_images:])

    return processed_results

SERPER_BASE_URL = os.getenv("SERPER_BASE_URL", "https://google.serper.dev")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# Initialize FastMCP server
mcp = FastMCP("serper-mcp-server")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(
        (requests.ConnectionError, requests.Timeout, requests.HTTPError)
    ),
    before_sleep=lambda retry_state: logger.warning(
        f"Serper API request failed (attempt {retry_state.attempt_number}), "
        f"retrying in {retry_state.next_action.sleep:.1f}s: {retry_state.outcome.exception()}"
    ),
)
def make_serper_request(
    endpoint: str, payload: Dict[str, Any], headers: Dict[str, str]
) -> requests.Response:
    """Make HTTP request to Serper API with retry logic."""
    response = requests.post(
        f"{SERPER_BASE_URL}/{endpoint}", json=payload, headers=headers, timeout=30
    )
    response.raise_for_status()
    return response


def _is_huggingface_dataset_or_space_url(url):
    """
    Check if the URL is a HuggingFace dataset or space URL.
    :param url: The URL to check
    :return: True if it's a HuggingFace dataset or space URL, False otherwise
    """
    if not url:
        return False
    return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url


@mcp.tool()
def google_search(
    q: str,
    gl: str = "us",
    hl: str = "en",
    location: str | None = None,
    num: int | None = None,
    tbs: str | None = None,
    page: int | None = None,
    autocorrect: bool | None = None,
):
    """
    Tool to perform web searches via Serper API and retrieve rich results.

    It is able to retrieve organic search results, people also ask,
    related searches, and knowledge graph.

    Args:
        q: Search query string
        gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
        hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
        location: Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')
        num: Number of results to return (default: 10)
        tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week,
            'qdr:m' for past month, 'qdr:y' for past year)
        page: Page number of results to return (default: 1)
        autocorrect: Whether to autocorrect spelling in query

    Returns:
        Dictionary containing search results and metadata.
    """
    # Validate required parameter
    if not q or not q.strip():
        return json.dumps(
            {
                "success": False,
                "error": "Search query 'q' is required and cannot be empty",
                "results": [],
            },
            ensure_ascii=False,
        )

    try:
        if not SERPER_API_KEY:
            return json.dumps(
                {"success": False, "error": "SERPER_API_KEY environment variable not set", "results": []},
                ensure_ascii=False,
            )

        payload: dict[str, Any] = {"q": q.strip(), "gl": gl, "hl": hl}
        if location:
            payload["location"] = location
        payload["num"] = num if num is not None else 10
        if tbs:
            payload["tbs"] = tbs
        if page is not None:
            payload["page"] = page
        if autocorrect is not None:
            payload["autocorrect"] = autocorrect

        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        response = make_serper_request("search", payload, headers)
        data = response.json()

        # filter out HuggingFace dataset or space urls
        organic_results = []
        if "organic" in data:
            for item in data["organic"]:
                if _is_huggingface_dataset_or_space_url(item.get("link", "")):
                    continue
                organic_results.append(item)

        # Limit organic results to the requested number
        requested_num = num if num is not None else 10
        organic_results = organic_results[:requested_num]

        # Keep all original fields, but overwrite "organic"
        response_data = dict(data)
        response_data["organic"] = organic_results
        response_data = decode_http_urls_in_dict(response_data)

        result_json = json.dumps(response_data, ensure_ascii=False)

        return result_json
    except RetryError as e:
        last_exception = e.last_attempt.exception()
        status_code = None
        if isinstance(last_exception, requests.HTTPError) and last_exception.response is not None:
            status_code = last_exception.response.status_code
        error_msg = (
            f"Serper API request failed after retries: "
            f"status_code={status_code}, last_error={str(last_exception)}"
        )
        logger.error(error_msg)
        return json.dumps(
            {"success": False, "retryable": True, "error": error_msg, "results": []},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Unexpected error in google_search: {str(e)}")
        return json.dumps(
            {"success": False, "error": f"Unexpected error: {str(e)}", "results": []},
            ensure_ascii=False,
        )


@mcp.tool()
def scholar_search(
    q: str,
    gl: str = "us",
    hl: str = "en",
    num: int | None = None,
    page: int | None = None,
):
    """
    Tool to perform academic searches via Google Scholar through Serper API.

    Retrieve scholarly literature including articles, theses, books,
    abstracts, and court opinions from academic publishers, professional
    societies, online repositories, and universities.

    Args:
        q: Search query string for academic literature
        gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
        hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
        num: Number of results to return (default: 10)
        page: Page number of results to return (default: 1)

    Returns:
        Dictionary containing scholarly search results and metadata.
    """
    # Validate required parameter
    if not q or not q.strip():
        return json.dumps(
            {
                "success": False,
                "error": "Search query 'q' is required and cannot be empty",
                "results": [],
            },
            ensure_ascii=False,
        )

    try:
        requested_num = num if num is not None else 10

        if not SERPER_API_KEY:
            return json.dumps(
                {"success": False, "error": "SERPER_API_KEY environment variable not set", "results": []},
                ensure_ascii=False,
            )

        payload: dict[str, Any] = {"q": q.strip(), "gl": gl, "hl": hl}
        payload["num"] = requested_num
        if page is not None:
            payload["page"] = page

        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        response = make_serper_request("scholar", payload, headers)
        data = response.json()

        data = decode_http_urls_in_dict(data)

        if "organic" in data and isinstance(data["organic"], list):
            data["organic"] = data["organic"][:requested_num]

        return json.dumps(data, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Unexpected error: {str(e)}", "results": []},
            ensure_ascii=False,
        )


@mcp.tool()
def image_search(
    q: str,
    gl: str = "us",
    hl: str = "en",
    location: str | None = None,
    num: int | None = None,
    page: int | None = None,
):
    """
    Tool to perform image searches via Serper API and retrieve visual results.

    Retrieve image search results including thumbnails, titles, and source URLs.
    Returns image metadata with URLs for reference, without downloading images.

    Args:
        q: Search query string for images
        gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
        hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
        location: Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')
        num: Number of results to return (default: 5)
        page: Page number of results to return (default: 1)

    Returns:
        Dictionary containing image search results and metadata.
        Images are returned with URLs and metadata, without base64 encoding.
    """
    # Validate required parameter
    if not q or not q.strip():
        return json.dumps(
            {"success": False, "error": "Search query 'q' is required and cannot be empty", "results": []},
            ensure_ascii=False,
        )

    try:
        if not SERPER_API_KEY:
            return json.dumps(
                {"success": False, "error": "SERPER_API_KEY environment variable not set", "results": []},
                ensure_ascii=False,
            )

        payload: dict[str, Any] = {"q": q.strip(), "gl": gl, "hl": hl}
        if location:
            payload["location"] = location
        payload["num"] = num if num is not None else 5
        if page is not None:
            payload["page"] = page

        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        response = make_serper_request("images", payload, headers)
        data = response.json()

        data = decode_http_urls_in_dict(data)

        requested_num = num if num is not None else 5
        if "images" in data and isinstance(data["images"], list):
            data["images"] = data["images"][:requested_num]

        result_json = json.dumps(data, ensure_ascii=False)

        return result_json

    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Unexpected error: {str(e)}", "results": []},
            ensure_ascii=False,
        )


@mcp.tool()
def visual_search(
    image_url: str,
    gl: str = "us",
    hl: str = "en",
    location: str | None = None,
    num: int | None = None,
    page: int | None = None,
):
    """
    Tool to perform visual searches via Serper Lens API to find similar images.

    Given an image URL, retrieve visually similar images from across the web.
    Returns image metadata with URLs for reference, without downloading images.

    Args:
        image_url: URL of the image to search with
        gl: Optional region code for search results in ISO 3166-1 alpha-2 format (e.g., 'us')
        hl: Optional language code for search results in ISO 639-1 format (e.g., 'en')
        location: Optional location for search results (e.g., 'SoHo, New York, United States', 'California, United States')
        num: Number of results to return (default: 5)
        page: Page number of results to return (default: 1)

    Returns:
        Dictionary containing visually similar image search results and metadata.
        Images are returned with URLs and metadata, without base64 encoding.
    """
    # Validate required parameter
    if not image_url or not image_url.strip():
        return json.dumps(
            {"success": False, "error": "Image URL 'image_url' is required and cannot be empty", "results": []},
            ensure_ascii=False,
        )

    # Basic URL validation
    if not image_url.startswith(("http://", "https://")):
        return json.dumps(
            {"success": False, "error": "Invalid image URL format. URLs must start with http:// or https://", "results": []},
            ensure_ascii=False,
        )

    try:
        requested_num = num if num is not None else 5

        if not SERPER_API_KEY:
            return json.dumps(
                {"success": False, "error": "SERPER_API_KEY environment variable not set", "results": []},
                ensure_ascii=False,
            )

        payload: dict[str, Any] = {"url": image_url.strip(), "gl": gl, "hl": hl}
        if location:
            payload["location"] = location
        payload["num"] = requested_num
        if page is not None:
            payload["page"] = page

        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        response = make_serper_request("lens", payload, headers)
        data = response.json()

        data = decode_http_urls_in_dict(data)

        # Limit organic results to the requested number
        requested_num = num if num is not None else 5
        if "organic" in data and isinstance(data["organic"], list):
            data["organic"] = data["organic"][:requested_num]

        # Limit images to requested number (return metadata only, no download/encoding)
        if "images" in data and isinstance(data["images"], list):
            data["images"] = data["images"][:requested_num]

        return json.dumps(data, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Unexpected error: {str(e)}", "results": []},
            ensure_ascii=False,
        )


if __name__ == "__main__":
    mcp.run()
