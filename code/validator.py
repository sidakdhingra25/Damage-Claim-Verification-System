"""Deterministic post-vision consistency validation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from evidence import get_evidence_standard_met

MISMATCH_RISK_FLAGS = frozenset({"wrong_object", "claim_mismatch"})


def _normalize_risk_flags(risk_flags: Any) -> list[str]:
    if isinstance(risk_flags, str):
        parts = [part.strip() for part in risk_flags.split(";")]
    elif isinstance(risk_flags, list):
        parts = [str(part).strip() for part in risk_flags]
    else:
        parts = []

    return [flag for flag in parts if flag and flag.lower() != "none"]


def _normalize_supporting_image_ids(supporting_image_ids: Any) -> list[str]:
    if isinstance(supporting_image_ids, str):
        parts = [part.strip() for part in supporting_image_ids.split(";")]
    elif isinstance(supporting_image_ids, list):
        parts = [str(part).strip() for part in supporting_image_ids]
    else:
        parts = []

    return [image_id for image_id in parts if image_id and image_id.lower() != "none"]


def _has_supporting_images(supporting_image_ids: list[str]) -> bool:
    return bool(supporting_image_ids)


def validate(
    extraction: dict[str, Any],
    history_flags: list[str],
    *,
    claim_object: str,
    requirements_df: pd.DataFrame,
    num_images: int,
) -> tuple[dict[str, Any], list[str]]:
    """Validate VisionExtraction, evaluate evidence, and return output-ready result."""
    fired_rules: list[str] = []
    risk_flags = _normalize_risk_flags(extraction.get("risk_flags", []))
    normalized_history_flags = _normalize_risk_flags(history_flags)
    valid_image = bool(extraction.get("valid_image"))
    issue_type = str(extraction.get("issue_type", "unknown"))
    object_part = str(extraction.get("object_part", "unknown"))
    severity = str(extraction.get("severity", "unknown"))
    supporting_image_ids = _normalize_supporting_image_ids(
        extraction.get("supporting_image_ids", [])
    )
    visual_claim_status = str(
        extraction.get("visual_claim_status", "not_enough_information")
    )
    claim_status_justification = str(
        extraction.get("claim_status_justification", "")
    )

    if not valid_image:
        evidence_standard_met = False
        evidence_standard_met_reason = (
            "Submitted images are not usable enough to evaluate the claim."
        )
    elif MISMATCH_RISK_FLAGS.intersection(risk_flags):
        evidence_standard_met = False
        evidence_standard_met_reason = (
            "Submitted images do not consistently show the same claimed object."
        )
    else:
        evidence_standard_met, evidence_standard_met_reason = get_evidence_standard_met(
            issue_type,
            object_part,
            claim_object,
            requirements_df,
            num_images,
            supporting_image_ids,
        )

    claim_status = visual_claim_status
    if not evidence_standard_met and visual_claim_status == "supported":
        claim_status = "not_enough_information"
        fired_rules.append("insufficient_evidence_overrides_supported")

    if "text_instruction_present" in risk_flags:
        fired_rules.append("text_instruction_present_preserved")

    if issue_type == "none" and severity != "none":
        severity = "none"
        fired_rules.append("issue_type_none_forces_severity_none")

    if "wrong_object" in risk_flags and claim_status == "supported":
        if evidence_standard_met is False:
            claim_status = "not_enough_information"
        else:
            claim_status = "contradicted"
        fired_rules.append("wrong_object_cannot_be_supported")

    if (
        "user_history_risk" in normalized_history_flags
        or MISMATCH_RISK_FLAGS.intersection(risk_flags)
    ):
        if "manual_review_required" not in risk_flags:
            risk_flags.append("manual_review_required")
        fired_rules.append("manual_review_required_from_risk_correlation")

    if claim_status == "supported" and not _has_supporting_images(supporting_image_ids):
        claim_status = "not_enough_information"
        if "manual_review_required" not in risk_flags:
            risk_flags.append("manual_review_required")
        fired_rules.append("supported_with_no_supporting_images")

    result = {
        "conversation_injection_detected": "text_instruction_present" in risk_flags,
        "per_image": [],
        "aggregated": {
            "valid_image": valid_image,
            "evidence_standard_met": evidence_standard_met,
            "evidence_standard_met_reason": evidence_standard_met_reason,
            "issue_type": issue_type,
            "object_part": object_part,
            "claim_status": claim_status,
            "claim_status_justification": claim_status_justification,
            "supporting_image_ids": supporting_image_ids,
            "severity": severity,
            "risk_flags": risk_flags,
        },
    }
    return result, fired_rules
