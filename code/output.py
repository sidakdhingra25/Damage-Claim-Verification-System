"""Format pipeline results into output.csv rows."""

from __future__ import annotations

from typing import Any

OUTPUT_COLUMNS: tuple[str, ...] = (
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
)


def _bool_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized
    return "true" if bool(value) else "false"


def _normalize_list_field(values: Any) -> list[str]:
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(";")]
    elif isinstance(values, list):
        parts = [str(part).strip() for part in values]
    else:
        parts = []

    return [part for part in parts if part and part.lower() != "none"]


def _join_or_none(values: Any) -> str:
    parts = _normalize_list_field(values)
    return ";".join(parts) if parts else "none"


def _merge_risk_flags(vision_flags: Any, history_flags: list[str]) -> str:
    merged: list[str] = []

    for flag in _normalize_list_field(vision_flags) + _normalize_list_field(history_flags):
        if flag not in merged:
            merged.append(flag)

    return ";".join(merged) if merged else "none"


def format_row(
    claim_row: Any,
    vision_result: dict[str, Any],
    history_flags: list[str],
    validated_result: dict[str, Any],
) -> dict[str, str]:
    """Build one output.csv row with the exact required column order."""
    _ = vision_result

    aggregated = validated_result.get("aggregated", {})

    row = {
        "user_id": str(claim_row["user_id"]),
        "image_paths": str(claim_row["image_paths"]),
        "user_claim": str(claim_row["user_claim"]),
        "claim_object": str(claim_row["claim_object"]),
        "evidence_standard_met": _bool_string(aggregated.get("evidence_standard_met")),
        "evidence_standard_met_reason": str(
            aggregated.get("evidence_standard_met_reason", "")
        ),
        "risk_flags": _merge_risk_flags(aggregated.get("risk_flags", []), history_flags),
        "issue_type": str(aggregated.get("issue_type", "")),
        "object_part": str(aggregated.get("object_part", "")),
        "claim_status": str(aggregated.get("claim_status", "")),
        "claim_status_justification": str(
            aggregated.get("claim_status_justification", "")
        ),
        "supporting_image_ids": _join_or_none(aggregated.get("supporting_image_ids", [])),
        "valid_image": _bool_string(aggregated.get("valid_image")),
        "severity": str(aggregated.get("severity", "")),
    }

    row["issue_type"] = str(row.get("issue_type", "")).strip().lower()
    row["claim_status"] = str(row.get("claim_status", "")).strip().lower()
    row["severity"] = str(row.get("severity", "")).strip().lower()

    normalized_risk_flags = sorted(_normalize_list_field(row.get("risk_flags", "")))
    row["risk_flags"] = (
        ";".join(normalized_risk_flags) if normalized_risk_flags else "none"
    )

    return {column: row[column] for column in OUTPUT_COLUMNS}
