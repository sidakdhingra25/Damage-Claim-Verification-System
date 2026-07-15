"""Deterministic evidence requirement lookup and evaluation."""

from __future__ import annotations

import pandas as pd

ISSUE_TYPE_TO_FAMILY: dict[str, str] = {
    "dent": "dent_or_scratch",
    "scratch": "dent_or_scratch",
    "crack": "breakage",
    "glass_shatter": "breakage",
    "broken_part": "breakage",
    "missing_part": "breakage",
    "water_damage": "stain",
    "stain": "stain",
    "torn_packaging": "package_exterior",
    "crushed_packaging": "package_exterior",
    "none": "general",
    "unknown": "general",
}

CAR_GLASS_LIGHT_MIRROR_PARTS = frozenset(
    {"windshield", "headlight", "taillight", "side_mirror"}
)
LAPTOP_SCREEN_KEYBOARD_TRACKPAD_PARTS = frozenset(
    {"screen", "keyboard", "trackpad"}
)

FAMILY_REQUIREMENT_IDS: dict[tuple[str, str], list[str]] = {
    ("dent_or_scratch", "car"): ["REQ_CAR_BODY_PANEL"],
    ("dent_or_scratch", "laptop"): ["REQ_LAPTOP_BODY_HINGE_PORT"],
    ("dent_or_scratch", "package"): ["REQ_GENERAL_OBJECT_PART"],
    ("breakage", "car"): ["REQ_CAR_GLASS_LIGHT_MIRROR"],
    ("breakage", "laptop"): ["REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD"],
    ("breakage", "package"): ["REQ_PACKAGE_CONTENTS"],
    ("stain", "car"): ["REQ_GENERAL_OBJECT_PART"],
    ("stain", "laptop"): ["REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD"],
    ("stain", "package"): ["REQ_PACKAGE_LABEL_OR_STAIN"],
    ("package_exterior", "package"): ["REQ_PACKAGE_EXTERIOR"],
    ("contents", "package"): ["REQ_PACKAGE_CONTENTS"],
    ("general", "car"): ["REQ_GENERAL_OBJECT_PART", "REQ_REVIEW_TRUST"],
    ("general", "laptop"): ["REQ_GENERAL_OBJECT_PART", "REQ_REVIEW_TRUST"],
    ("general", "package"): ["REQ_GENERAL_OBJECT_PART", "REQ_REVIEW_TRUST"],
}

ALWAYS_REQUIREMENT_IDS = ["REQ_REVIEW_TRUST"]
MULTI_IMAGE_REQUIREMENT_ID = "REQ_GENERAL_MULTI_IMAGE"

REQUIREMENT_MIN_SUPPORTING_IMAGES: dict[str, int] = {
    "REQ_GENERAL_OBJECT_PART": 1,
    "REQ_GENERAL_MULTI_IMAGE": 1,
    "REQ_CAR_BODY_PANEL": 1,
    "REQ_CAR_GLASS_LIGHT_MIRROR": 1,
    "REQ_CAR_IDENTITY_OR_SIDE": 2,
    "REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD": 1,
    "REQ_LAPTOP_BODY_HINGE_PORT": 1,
    "REQ_PACKAGE_EXTERIOR": 1,
    "REQ_PACKAGE_LABEL_OR_STAIN": 1,
    "REQ_PACKAGE_CONTENTS": 1,
    "REQ_REVIEW_TRUST": 1,
}


def _minimum_image_count(rule: dict[str, str | int]) -> int:
    raw = rule.get("minimum_image_evidence")
    if isinstance(raw, int):
        return max(0, raw)
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())

    requirement_id = str(rule.get("requirement_id", ""))
    return REQUIREMENT_MIN_SUPPORTING_IMAGES.get(requirement_id, 1)


def _normalize_supporting_image_ids(supporting_image_ids: list[str]) -> list[str]:
    return [
        image_id
        for image_id in supporting_image_ids
        if str(image_id).strip() and str(image_id).strip().lower() != "none"
    ]


def evaluate_evidence_standard(
    matched_rules: list[dict[str, str | int]],
    supporting_image_ids: list[str],
) -> tuple[bool, str]:
    """Check whether supporting images satisfy matched rule minimum counts."""
    supporting = _normalize_supporting_image_ids(supporting_image_ids)
    provided_count = len(supporting)

    if not matched_rules:
        return True, (
            f"No matched evidence rules; {provided_count} supporting image(s) accepted."
        )

    failures: list[str] = []
    for rule in matched_rules:
        required_count = _minimum_image_count(rule)
        requirement_id = str(rule.get("requirement_id", "unknown"))
        if provided_count < required_count:
            description = str(
                rule.get(
                    "minimum_image_evidence_description",
                    rule.get("minimum_image_evidence", ""),
                )
            )
            failures.append(
                f"{requirement_id} requires at least {required_count} supporting "
                f"image(s), but {provided_count} provided"
                + (f" ({description})" if description and not description.isdigit() else "")
                + "."
            )

    if failures:
        return False, " ".join(failures)

    required_counts = [_minimum_image_count(rule) for rule in matched_rules]
    highest_required = max(required_counts)
    matched_ids = ", ".join(str(rule.get("requirement_id", "unknown")) for rule in matched_rules)
    return True, (
        f"Supporting image count ({provided_count}) meets or exceeds all matched "
        f"evidence minimums (highest required: {highest_required}) for "
        f"{matched_ids}."
    )


def resolve_issue_family(issue_type_guess: str, claim_object: str) -> str:
    if issue_type_guess == "missing_part" and claim_object == "package":
        return "contents"
    return ISSUE_TYPE_TO_FAMILY.get(issue_type_guess, "general")


def _refine_requirement_ids(
    requirement_ids: list[str],
    family: str,
    claim_object: str,
    issue_type_guess: str,
    object_part_guess: str,
) -> list[str]:
    refined = list(requirement_ids)

    if family == "breakage" and claim_object == "car":
        if (
            object_part_guess in CAR_GLASS_LIGHT_MIRROR_PARTS
            or issue_type_guess in {"crack", "glass_shatter"}
        ):
            refined = ["REQ_CAR_GLASS_LIGHT_MIRROR"]
        elif object_part_guess not in {"", "unknown", "body"}:
            refined = ["REQ_CAR_BODY_PANEL"]

    if family == "breakage" and claim_object == "laptop":
        if object_part_guess in LAPTOP_SCREEN_KEYBOARD_TRACKPAD_PARTS:
            refined = ["REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD"]
        elif object_part_guess not in {"", "unknown"}:
            refined = ["REQ_LAPTOP_BODY_HINGE_PORT"]

    return refined


def resolve_requirements(
    family: str,
    claim_object: str,
    requirements_df: pd.DataFrame,
    *,
    issue_type_guess: str = "",
    object_part_guess: str = "",
    num_images: int = 1,
) -> list[dict[str, str]]:
    """Return matched requirement rows as plain dicts (not a DataFrame)."""
    requirement_ids = list(
        FAMILY_REQUIREMENT_IDS.get(
            (family, claim_object),
            FAMILY_REQUIREMENT_IDS.get(
                ("general", claim_object),
                ["REQ_GENERAL_OBJECT_PART", "REQ_REVIEW_TRUST"],
            ),
        )
    )
    requirement_ids = _refine_requirement_ids(
        requirement_ids,
        family,
        claim_object,
        issue_type_guess,
        object_part_guess,
    )

    ordered_ids: list[str] = []
    for req_id in requirement_ids + ALWAYS_REQUIREMENT_IDS:
        if req_id not in ordered_ids:
            ordered_ids.append(req_id)

    if num_images >= 2 and MULTI_IMAGE_REQUIREMENT_ID not in ordered_ids:
        ordered_ids.append(MULTI_IMAGE_REQUIREMENT_ID)

    matched: list[dict[str, str]] = []
    for req_id in ordered_ids:
        rows = requirements_df.loc[requirements_df["requirement_id"] == req_id]
        if rows.empty:
            matched.append(
                {
                    "requirement_id": req_id,
                    "claim_object": "",
                    "applies_to": "",
                    "minimum_image_evidence": REQUIREMENT_MIN_SUPPORTING_IMAGES.get(req_id, 1),
                    "minimum_image_evidence_description": (
                        "Requirement not found in evidence_requirements.csv."
                    ),
                }
            )
            continue

        row = rows.iloc[0]
        matched.append(
            {
                "requirement_id": str(row["requirement_id"]),
                "claim_object": str(row["claim_object"]),
                "applies_to": str(row["applies_to"]),
                "minimum_image_evidence": REQUIREMENT_MIN_SUPPORTING_IMAGES.get(req_id, 1),
                "minimum_image_evidence_description": str(row["minimum_image_evidence"]),
            }
        )

    return matched


def check_evidence(
    issue_type_guess: str,
    object_part_guess: str,
    claim_object: str,
    requirements_df: pd.DataFrame,
    num_images: int,
) -> str:
    """Return readable evidence requirement text for the vision model tool."""
    family = resolve_issue_family(issue_type_guess, claim_object)
    matched = resolve_requirements(
        family,
        claim_object,
        requirements_df,
        issue_type_guess=issue_type_guess,
        object_part_guess=object_part_guess,
        num_images=num_images,
    )

    lines = [
        "Evidence requirements for this claim:",
        f"- Claim object: {claim_object}",
        f"- Issue type (guess): {issue_type_guess}",
        f"- Object part (guess): {object_part_guess or 'unknown'}",
        f"- Evidence family: {family}",
        f"- Number of images: {num_images}",
        "",
        "Matched requirements:",
    ]

    for index, requirement in enumerate(matched, start=1):
        lines.extend(
            [
                (
                    f"{index}. [{requirement['requirement_id']}] "
                    f"(applies_to: {requirement['applies_to'] or 'n/a'})"
                ),
                f"   {requirement['minimum_image_evidence_description']}",
            ]
        )

    return "\n".join(lines)


def get_evidence_standard_met(
    issue_type: str,
    object_part: str,
    claim_object: str,
    requirements_df: pd.DataFrame,
    num_images: int,
    supporting_image_ids: list[str],
) -> tuple[bool, str]:
    """Resolve matched rules from issue_type/object_part and evaluate supporting images."""
    family = resolve_issue_family(issue_type, claim_object)
    matched_rules = resolve_requirements(
        family,
        claim_object,
        requirements_df,
        issue_type_guess=issue_type,
        object_part_guess=object_part,
        num_images=num_images,
    )
    return evaluate_evidence_standard(matched_rules, supporting_image_ids)
