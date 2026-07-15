"""Compare sample predictions against labeled sample claims."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from output import OUTPUT_COLUMNS  # noqa: E402

PREDICTION_FIELDS: tuple[str, ...] = OUTPUT_COLUMNS[4:]
LIST_FIELDS = frozenset({"risk_flags", "supporting_image_ids"})
BOOL_FIELDS = frozenset({"evidence_standard_met", "valid_image"})

PREDICTIONS_PATH = EVAL_DIR / "sample_predictions.csv"
GROUND_TRUTH_PATH = REPO_ROOT / "dataset" / "sample_claims.csv"
DETAIL_PATH = EVAL_DIR / "evaluation_detail.csv"


def _normalize_list_field(value: str) -> str:
    parts = [
        part.strip()
        for part in str(value).split(";")
        if part.strip() and part.strip().lower() != "none"
    ]
    return ";".join(sorted(parts)) if parts else "none"


def normalize_field(field: str, value: str) -> str:
    text = str(value).strip()
    if field in LIST_FIELDS:
        return _normalize_list_field(text)
    if field in BOOL_FIELDS:
        return text.lower()
    return text


def fields_match(field: str, expected: str, predicted: str) -> bool:
    return normalize_field(field, expected) == normalize_field(field, predicted)


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def _build_detail_row(
    user_id: str,
    mismatch_count: int,
    mismatched_fields: list[str],
    expected_row: pd.Series,
    predicted_row: pd.Series,
) -> dict[str, str]:
    detail: dict[str, str] = {
        "user_id": user_id,
        "mismatch_count": str(mismatch_count),
        "mismatched_fields": ";".join(mismatched_fields),
    }
    for field in PREDICTION_FIELDS:
        detail[f"{field}_expected"] = str(expected_row[field])
        detail[f"{field}_predicted"] = str(predicted_row[field])
    return detail


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        print(f"Missing predictions file: {PREDICTIONS_PATH}", file=sys.stderr)
        sys.exit(1)
    if not GROUND_TRUTH_PATH.exists():
        print(f"Missing ground truth file: {GROUND_TRUTH_PATH}", file=sys.stderr)
        sys.exit(1)

    expected_df = _load_csv(GROUND_TRUTH_PATH)
    predicted_df = _load_csv(PREDICTIONS_PATH)

    missing_columns = [column for column in OUTPUT_COLUMNS if column not in expected_df.columns]
    if missing_columns:
        print(f"Ground truth missing columns: {missing_columns}", file=sys.stderr)
        sys.exit(1)

    missing_columns = [column for column in OUTPUT_COLUMNS if column not in predicted_df.columns]
    if missing_columns:
        print(f"Predictions missing columns: {missing_columns}", file=sys.stderr)
        sys.exit(1)

    expected_df = expected_df[list(OUTPUT_COLUMNS)].copy()
    predicted_df = predicted_df[list(OUTPUT_COLUMNS)].copy()

    if len(expected_df) != len(predicted_df):
        print(
            f"Row count mismatch: expected={len(expected_df)} predicted={len(predicted_df)}",
            file=sys.stderr,
        )
        sys.exit(1)

    expected_ids = expected_df["user_id"].tolist()
    predicted_ids = predicted_df["user_id"].tolist()
    if expected_ids != predicted_ids:
        print("Warning: row order/user_id sequence differs; aligning on user_id.", file=sys.stderr)
        predicted_df = predicted_df.set_index("user_id")
        expected_df = expected_df.set_index("user_id")
        missing_users = expected_df.index.difference(predicted_df.index)
        extra_users = predicted_df.index.difference(expected_df.index)
        if len(missing_users) or len(extra_users):
            print(f"Missing predictions for: {list(missing_users)}", file=sys.stderr)
            print(f"Unexpected predictions for: {list(extra_users)}", file=sys.stderr)
            sys.exit(1)
        predicted_df = predicted_df.loc[expected_df.index].reset_index()
        expected_df = expected_df.reset_index()

    total_rows = len(expected_df)
    field_matches = {field: 0 for field in PREDICTION_FIELDS}
    exact_matches = 0
    detail_rows: list[dict[str, str]] = []

    for row_index in range(total_rows):
        expected_row = expected_df.iloc[row_index]
        predicted_row = predicted_df.iloc[row_index]
        user_id = str(expected_row["user_id"])

        mismatched_fields: list[str] = []
        for field in PREDICTION_FIELDS:
            if fields_match(field, expected_row[field], predicted_row[field]):
                field_matches[field] += 1
            else:
                mismatched_fields.append(field)

        if not mismatched_fields:
            exact_matches += 1
        else:
            detail_rows.append(
                _build_detail_row(
                    user_id,
                    len(mismatched_fields),
                    mismatched_fields,
                    expected_row,
                    predicted_row,
                )
            )

    detail_rows.sort(
        key=lambda row: (-int(row["mismatch_count"]), row["user_id"]),
    )

    detail_columns = ["user_id", "mismatch_count", "mismatched_fields"]
    for field in PREDICTION_FIELDS:
        detail_columns.extend([f"{field}_expected", f"{field}_predicted"])

    detail_df = pd.DataFrame(detail_rows, columns=detail_columns)
    detail_df.to_csv(DETAIL_PATH, index=False, quoting=1)

    print(f"Compared {total_rows} rows")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Ground truth: {GROUND_TRUTH_PATH}")
    print()
    print("Per-field accuracy:")
    for field in PREDICTION_FIELDS:
        matches = field_matches[field]
        accuracy = (matches / total_rows) * 100 if total_rows else 0.0
        print(f"  {field}: {matches}/{total_rows} ({accuracy:.1f}%)")

    exact_rate = (exact_matches / total_rows) * 100 if total_rows else 0.0
    print()
    print(f"Exact-match rate: {exact_matches}/{total_rows} ({exact_rate:.1f}%)")
    print(f"Failing rows: {len(detail_rows)}")
    print(f"Detail CSV: {DETAIL_PATH}")


if __name__ == "__main__":
    main()
