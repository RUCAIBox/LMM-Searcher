# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""Output formatting utilities for agent responses."""

import base64
import json
import logging
import re
from typing import Optional, Tuple, Union

import requests

from miroflow_tools.image_llm_payload import ensure_image_base64_under_limit

from ..utils.image_utils import ensure_image_dimensions
from ..utils.prompt_utils import FORMAT_ERROR_MESSAGE

logger = logging.getLogger(__name__)

# Maximum length for tool results before truncation (100k chars ≈ 25k tokens)
TOOL_RESULT_MAX_LENGTH = 100_000


class OutputFormatter:
    """Formatter for processing and formatting agent outputs."""

    def _extract_boxed_content(self, text: str) -> str:
        r"""
        Extract the content of the last \boxed{...} occurrence in the given text.

        Supports:
          - Arbitrary levels of nested braces
          - Escaped braces (\{ and \})
          - Whitespace between \boxed and the opening brace
          - Empty content inside braces
          - Incomplete boxed expressions (extracts to end of string as fallback)

        Args:
            text: Input text that may contain \boxed{...} expressions

        Returns:
            The extracted boxed content, or empty string if no match is found.
        """
        if not text:
            return ""

        _BOXED_RE = re.compile(r"\\boxed\b", re.DOTALL)

        last_result = None  # Track the last boxed content (complete or incomplete)
        i = 0
        n = len(text)

        while True:
            # Find the next \boxed occurrence
            m = _BOXED_RE.search(text, i)
            if not m:
                break
            j = m.end()

            # Skip any whitespace after \boxed
            while j < n and text[j].isspace():
                j += 1

            # Require that the next character is '{'
            if j >= n or text[j] != "{":
                i = j
                continue

            # Parse the brace content manually to handle nesting and escapes
            depth = 0
            k = j
            escaped = False
            found_closing = False
            while k < n:
                ch = text[k]
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    # When depth returns to zero, the boxed content ends
                    if depth == 0:
                        last_result = text[j + 1 : k]
                        i = k + 1
                        found_closing = True
                        break
                k += 1

            # If we didn't find a closing brace, this is an incomplete boxed
            # Store it as the last result (will be overwritten if we find more boxed later)
            if not found_closing and depth > 0:
                last_result = text[j + 1 : n]
                i = k  # Continue from where we stopped
            elif not found_closing:
                i = j + 1  # Move past this invalid boxed

        # Return the last boxed content found (complete or incomplete)
        black_list = ["?", "??", "???", "？", "……", "…", "...", "unknown", None]
        return last_result.strip() if last_result not in black_list else ""

    _THUMBNAIL_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    _IMAGE_MAGIC_BYTES = [
        (b'\xFF\xD8\xFF', 'image/jpeg'),
        (b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A', 'image/png'),
        (b'GIF87a', 'image/gif'),
        (b'GIF89a', 'image/gif'),
        (b'BM', 'image/bmp'),
        (b'RIFF', 'image/webp'),
    ]

    _LLM_SUPPORTED_MIME_TYPES = {
        'image/png', 'image/jpeg', 'image/gif', 'image/webp',
    }

    def _validate_image_bytes(self, data: bytes, url: str) -> Optional[str]:
        """Validate downloaded bytes are a real image with LLM-supported format. Returns detected mime type or None."""
        if not data or len(data) < 8:
            logger.warning(f"Thumbnail too small ({len(data) if data else 0} bytes) from {url}")
            return None

        content_start = data[:100].strip().lower()
        if content_start.startswith((b'<!doctype', b'<html', b'<head', b'<?xml')):
            logger.warning(
                f"Thumbnail is HTML/XML, not an image: url={url}, "
                f"first_bytes={data[:80]!r}"
            )
            return None

        for sig, mime in self._IMAGE_MAGIC_BYTES:
            if data.startswith(sig):
                if mime not in self._LLM_SUPPORTED_MIME_TYPES:
                    logger.warning(f"Skipping unsupported image format {mime} from {url}")
                    return None
                return mime

        try:
            from PIL import Image
            from io import BytesIO
            img = Image.open(BytesIO(data))
            img.verify()
            mime = f"image/{img.format.lower()}" if img.format else "image/jpeg"
            if mime not in self._LLM_SUPPORTED_MIME_TYPES:
                logger.warning(f"Skipping unsupported image format {mime} (detected by PIL) from {url}")
                return None
            return mime
        except Exception as e:
            logger.warning(
                f"Thumbnail failed PIL validation: url={url}, size={len(data)}, "
                f"first_20_bytes={data[:20]!r}, error={e}"
            )
            return None

    def _download_thumbnail(self, url: str, timeout: int = 10, max_retries: int = 2) -> Optional[str]:
        """
        Download a thumbnail image and return it as a base64 data URL.

        Args:
            url: The thumbnail image URL to download.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts on failure.

        Returns:
            Base64 data URL string, or None if download fails.
        """
        if not url:
            return None
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url, timeout=timeout, headers=self._THUMBNAIL_HEADERS
                )
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                raw_bytes = response.content

                detected_mime = self._validate_image_bytes(raw_bytes, url)
                if detected_mime is None:
                    logger.warning(
                        f"Skipping invalid thumbnail: url={url}, "
                        f"content_type={content_type}, size={len(raw_bytes)}"
                    )
                    return None

                raw_bytes = ensure_image_dimensions(raw_bytes)
                raw_bytes, out_mime = ensure_image_base64_under_limit(
                    raw_bytes, mime_type=detected_mime
                )
                image_base64 = base64.b64encode(raw_bytes).decode("utf-8")
                return f"data:{out_mime};base64,{image_base64}"
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"Failed to download thumbnail from {url}: {e}")
        return None

    def format_tool_result_for_user(
        self, tool_call_execution_result: dict
    ) -> Union[dict, list]:
        """
        Format tool execution results to be fed back to LLM as user messages.

        Only includes necessary information (results or errors). Long results
        are truncated to TOOL_RESULT_MAX_LENGTH to prevent context overflow.

        For image search results, returns a multi-modal format with base64 images.

        Args:
            tool_call_execution_result: Dict containing server_name, tool_name,
                and either 'result' or 'error'.

        Returns:
            Dict with 'type' and 'text' keys, or a list containing mixed
            content (text + images) for multi-modal models.
        """
        server_name = tool_call_execution_result["server_name"]
        tool_name = tool_call_execution_result["tool_name"]

        if "error" in tool_call_execution_result:
            # Provide concise error information to LLM
            content = f"Tool call to {tool_name} on {server_name} failed. Error: {tool_call_execution_result['error']}"
            return {"type": "text", "text": content}
        elif "result" in tool_call_execution_result:
            result = tool_call_execution_result["result"]

            # Check if this is a fetch_image result (multi-modal format)
            if tool_name == "fetch_image":
                try:
                    # Parse JSON result
                    if isinstance(result, str):
                        data = json.loads(result)
                    else:
                        data = result

                    # Check if this contains an error
                    if isinstance(data, dict) and "error" in data:
                        return {"type": "text", "text": f"Image download failed: {data['error']}"}

                    # Check if this is a multi-modal content list
                    if isinstance(data, list):
                        # Validate the structure
                        valid_content = True
                        for item in data:
                            if not isinstance(item, dict) or "type" not in item:
                                valid_content = False
                                break

                        if valid_content:
                            # Return the multi-modal content list directly
                            return data
                except (json.JSONDecodeError, KeyError, TypeError):
                    # If parsing fails, treat as regular text
                    pass

            # Check if this is an image search or visual search result
            if tool_name in ["image_search", "visual_search"]:
                try:
                    if isinstance(result, str):
                        data = json.loads(result)
                    else:
                        data = result

                    has_images = "images" in data and isinstance(data["images"], list)
                    has_organic = "organic" in data and isinstance(data["organic"], list)
                    if has_images or has_organic:
                        return self._format_image_search_result(
                            data, tool_name, server_name
                        )
                except (json.JSONDecodeError, KeyError):
                    pass  # Fall through to regular text formatting

            # Check if this is an image processing tool result (multi-modal format)
            if tool_name in ["zoom_in", "rotate", "flip", "put_box"]:
                try:
                    # Parse JSON result
                    if isinstance(result, str):
                        data = json.loads(result)
                    else:
                        data = result

                    # Check if this contains an error
                    if isinstance(data, list) and len(data) > 0:
                        # Validate the structure as multi-modal content
                        valid_content = True
                        for item in data:
                            if not isinstance(item, dict) or "type" not in item:
                                valid_content = False
                                break

                        if valid_content:
                            # Return the multi-modal content list directly
                            return data
                except (json.JSONDecodeError, KeyError, TypeError):
                    # If parsing fails, treat as regular text
                    pass

            # Provide the original output result of the tool
            content = result
            # Truncate overly long results to prevent context overflow
            if len(content) > TOOL_RESULT_MAX_LENGTH:
                content = content[:TOOL_RESULT_MAX_LENGTH] + "\n... [Result truncated]"
            return {"type": "text", "text": content}
        else:
            content = f"Tool call to {tool_name} on {server_name} completed, but produced no specific output or result."
            return {"type": "text", "text": content}

    def _format_image_search_result(
        self, data: dict, tool_name: str, server_name: str
    ) -> list:
        """
        Format image/visual search results as multi-modal content with thumbnails.

        For each result item, downloads the thumbnail image from thumbnailUrl
        and places it immediately after the item's text description, enabling
        the model to visually inspect search results.

        Handles both image_search ("images" array) and visual_search ("organic" array).

        Args:
            data: Parsed JSON response from image/visual search
            tool_name: Name of the tool that was called
            server_name: Name of the MCP server

        Returns:
            List of content items (text + images) in OpenAI API format
        """
        content_items = []

        images = data.get("images", [])
        organic = data.get("organic", [])
        items = images if images else organic

        search_type = (
            "Visual search" if tool_name == "visual_search" else "Image search"
        )
        header = f"{search_type} completed on {server_name}. Found {len(items)} results.\n"
        content_items.append({"type": "text", "text": header})

        max_thumbnails = 5
        thumbnails_loaded = 0

        for idx, item in enumerate(items):
            title = item.get("title", "")
            link = item.get("link", "")
            image_url = item.get("imageUrl", "")
            thumbnail_url = item.get("thumbnailUrl", "")

            parts = [f"{idx + 1}."]
            if title:
                parts.append(f"Title: {title}")
            if image_url:
                parts.append(f"Image URL: {image_url}")
            if link:
                parts.append(f"Source: {link}")

            still_need_thumbnails = thumbnails_loaded < max_thumbnails

            if still_need_thumbnails and thumbnail_url:
                base64_data = self._download_thumbnail(thumbnail_url)
                if base64_data:
                    text_desc = " | ".join(parts)
                    content_items.append({"type": "text", "text": text_desc})
                    content_items.append({"type": "text", "text": "Thumbnail: "})
                    content_items.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_data},
                        }
                    )
                    thumbnails_loaded += 1
                    continue
                elif "base64_data" in item:
                    text_desc = " | ".join(parts)
                    content_items.append({"type": "text", "text": text_desc})
                    content_items.append({"type": "text", "text": "Thumbnail: "})
                    content_items.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": item["base64_data"]},
                        }
                    )
                    thumbnails_loaded += 1
                    continue

            if thumbnail_url:
                parts.append(f"Thumbnail URL: {thumbnail_url}")
            text_desc = " | ".join(parts)
            content_items.append({"type": "text", "text": text_desc})

        return content_items

    def format_final_summary_and_log(
        self, final_answer_text: str, client=None
    ) -> Tuple[str, str, str]:
        """
        Format final summary information, including answers and token statistics.

        Args:
            final_answer_text: The final answer text from the agent
            client: Optional LLM client for token usage statistics

        Returns:
            Tuple of (summary_text, boxed_result, usage_log)
        """
        summary_lines = []
        summary_lines.append("\n" + "=" * 30 + " Final Answer " + "=" * 30)
        summary_lines.append(final_answer_text)

        # Extract boxed result - find the last match using safer regex patterns
        boxed_result = self._extract_boxed_content(final_answer_text)

        # Add extracted result section
        summary_lines.append("\n" + "-" * 20 + " Extracted Result " + "-" * 20)

        if boxed_result:
            summary_lines.append(boxed_result)
        elif final_answer_text:
            summary_lines.append("No \\boxed{} content found. Use the entire answer text as the result.")
            # boxed_result = FORMAT_ERROR_MESSAGE
            boxed_result = final_answer_text

        # Token usage statistics and cost estimation - use client method
        if client and hasattr(client, "format_token_usage_summary"):
            token_summary_lines, log_string = client.format_token_usage_summary()
            summary_lines.extend(token_summary_lines)
        else:
            # If no client or client doesn't support it, use default format
            summary_lines.append("\n" + "-" * 20 + " Token Usage & Cost " + "-" * 20)
            summary_lines.append("Token usage information not available.")
            summary_lines.append("-" * (40 + len(" Token Usage & Cost ")))
            log_string = "Token usage information not available."

        return "\n".join(summary_lines), boxed_result, log_string
