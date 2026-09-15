"""Constants for the damage claim verification pipeline."""

import os
import sys

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    sys.exit(
        "GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is not set. "
        "Export it before running the pipeline, e.g. set GEMINI_API_KEY=your_key_here"
    )

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_IMAGES = 5
MAX_RETRIES = 2

CLAIM_OBJECTS = ("car", "laptop", "package")

CAR_OBJECT_PARTS = (
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "unknown",
)

LAPTOP_OBJECT_PARTS = (
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "body",
    "unknown",
)

PACKAGE_OBJECT_PARTS = (
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
)

OBJECT_PARTS_BY_CLAIM_OBJECT = {
    "car": CAR_OBJECT_PARTS,
    "laptop": LAPTOP_OBJECT_PARTS,
    "package": PACKAGE_OBJECT_PARTS,
}

ISSUE_TYPES = (
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
)

CLAIM_STATUSES = (
    "supported",
    "contradicted",
    "not_enough_information",
)

SEVERITIES = (
    "none",
    "low",
    "medium",
    "high",
    "unknown",
)

RISK_FLAGS = (
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
)
