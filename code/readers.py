"""CSV loading and image path / base64 helpers."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from config import MAX_IMAGES

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def load_claims(filename: str) -> pd.DataFrame:
    path = Path(filename)
    if not path.is_absolute():
        path = DATASET_DIR / filename
    return _read_csv(path)


def load_user_history() -> pd.DataFrame:
    return _read_csv(DATASET_DIR / "user_history.csv")


def load_evidence_requirements() -> pd.DataFrame:
    return _read_csv(DATASET_DIR / "evidence_requirements.csv")


def _resolve_image_path(path: str) -> Path:
    """Resolve a CSV image path to an absolute filesystem path."""
    cleaned = path.strip()
    candidate = Path(cleaned)
    if candidate.is_absolute():
        return candidate

    dataset_relative = DATASET_DIR / cleaned
    if dataset_relative.exists():
        return dataset_relative

    repo_relative = REPO_ROOT / cleaned
    if repo_relative.exists():
        return repo_relative

    return dataset_relative


def _detect_image_format(raw: bytes) -> str:
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        header_window = raw[4:32]
        if b"avif" in header_window or b"avis" in header_window:
            return "avif"
    return "unknown"


def _extension_format_hint(suffix: str) -> str | None:
    normalized = suffix.lower()
    if normalized in {".jpg", ".jpeg"}:
        return "jpeg"
    if normalized == ".png":
        return "png"
    if normalized == ".webp":
        return "webp"
    if normalized == ".gif":
        return "gif"
    if normalized == ".avif":
        return "avif"
    return None


def _open_image(raw: bytes, path: str, detected_format: str) -> Image.Image:
    try:
        image = Image.open(BytesIO(raw))
        image.load()
        return image
    except Exception as exc:
        raise RuntimeError(
            f"Failed to decode image '{path}' (sniffed format: {detected_format}): {exc}"
        ) from exc


def _render_png(image: Image.Image) -> bytes:
    output = image.copy()
    if output.mode not in ("RGB", "RGBA", "L"):
        output = output.convert("RGBA")
    buffer = BytesIO()
    output.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _convert_to_png(raw: bytes, path: str, detected_format: str) -> bytes:
    image = _open_image(raw, path, detected_format)
    return _render_png(image)


def _normalize_image_bytes(raw: bytes, detected_format: str, path: str) -> tuple[bytes, str]:
    if detected_format == "jpeg":
        return raw, "image/jpeg"
    if detected_format == "png":
        return raw, "image/png"
    return _convert_to_png(raw, path, detected_format), "image/png"


def parse_image_paths(image_paths: str, user_id: str | None = None) -> list[str]:
    paths = [part.strip() for part in image_paths.split(";") if part.strip()]
    if len(paths) > MAX_IMAGES:
        if user_id is not None:
            print(
                f"Warning: user_id={user_id} has {len(paths)} images; "
                f"using first {MAX_IMAGES}."
            )
        paths = paths[:MAX_IMAGES]
    return paths


def encode_image_base64(path: str) -> tuple[str, str]:
    resolved = _resolve_image_path(path)
    raw = resolved.read_bytes()
    detected_format = _detect_image_format(raw)

    extension_hint = _extension_format_hint(resolved.suffix)
    if extension_hint is not None and extension_hint != detected_format:
        pass  # extension mismatch is handled; no need to print

    image_bytes, mime_type = _normalize_image_bytes(raw, detected_format, path)
    b64_data = base64.b64encode(image_bytes).decode("ascii")
    return mime_type, b64_data
