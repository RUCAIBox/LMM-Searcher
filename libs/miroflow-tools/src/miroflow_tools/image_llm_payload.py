# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""Clamp image bytes so base64 payload stays within LLM context limits (e.g. after resize)."""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Max length of base64 *string* (ASCII) for one image in chat payloads (~500 KiB).
MAX_BASE64_CHARS_DEFAULT = 500 * 1024

_MIN_JPEG_SIDE = 28


def _mime_from_magic(data: bytes) -> Optional[str]:
    if not data or len(data) < 12:
        return None
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _pil_to_rgb_for_jpeg(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[-1])
        else:
            background.paste(img)
        return background
    if img.mode == "P":
        if "transparency" in img.info:
            return _pil_to_rgb_for_jpeg(img.convert("RGBA"))
        return img.convert("RGB")
    return img.convert("RGB")


def _jpeg_bytes_under_base64_cap(
    rgb: Image.Image, max_base64_chars: int, min_side: int = _MIN_JPEG_SIDE
) -> bytes:
    """Re-encode as JPEG, lowering quality and scale until base64 length fits or best effort."""
    qualities = [90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30]
    w0, h0 = rgb.size
    scale = 1.0
    best = b""
    best_b64_len = float("inf")
    base_rgb = rgb
    for _ in range(24):
        if scale < 1.0:
            nw = max(1, int(w0 * scale))
            nh = max(1, int(h0 * scale))
            work = base_rgb.resize((nw, nh), Image.LANCZOS)
        else:
            work = base_rgb
        for q in qualities:
            buf = io.BytesIO()
            work.save(buf, format="JPEG", quality=q, optimize=True)
            cand = buf.getvalue()
            enc_len = len(base64.b64encode(cand))
            if enc_len < best_b64_len:
                best, best_b64_len = cand, enc_len
            if enc_len <= max_base64_chars:
                return cand
        if min(work.size) <= min_side:
            break
        scale *= 0.85
    if best:
        logger.warning(
            "JPEG re-encode could not fit base64 under %s chars; returning smallest "
            "attempt (~%s chars)",
            max_base64_chars,
            int(best_b64_len),
        )
        return best
    buf = io.BytesIO()
    base_rgb.save(buf, format="JPEG", quality=30, optimize=True)
    return buf.getvalue()


def ensure_image_base64_under_limit(
    image_bytes: bytes,
    *,
    mime_type: Optional[str] = None,
    max_base64_chars: int = MAX_BASE64_CHARS_DEFAULT,
) -> Tuple[bytes, str]:
    """
    If base64(image_bytes) exceeds *max_base64_chars*, convert to JPEG (quality + scale down)
    until under the limit or best effort.

    Call after dimension clamping (e.g. ensure_image_dimensions). Does not resize by itself.

    Returns:
        (final_bytes, mime_type) — mime_type is image/jpeg when re-encoded, else sniffed or *mime_type*.
    """
    if not image_bytes:
        return image_bytes, mime_type or "image/jpeg"

    b64_len = len(base64.b64encode(image_bytes))
    if b64_len <= max_base64_chars:
        out_mime = _mime_from_magic(image_bytes) or mime_type or "image/jpeg"
        return image_bytes, out_mime

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        logger.warning("Cannot open image for JPEG fallback: %s", e)
        return image_bytes, _mime_from_magic(image_bytes) or mime_type or "image/jpeg"

    rgb = _pil_to_rgb_for_jpeg(img)
    jpeg_bytes = _jpeg_bytes_under_base64_cap(rgb, max_base64_chars)
    return jpeg_bytes, "image/jpeg"
