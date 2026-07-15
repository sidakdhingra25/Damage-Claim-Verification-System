"""CLI entry point for the damage claim verification pipeline."""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent


def _load_dotenv() -> None:
    env_path = CODE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

import pandas as pd  # noqa: E402

from history import get_history_risk_flags  # noqa: E402
from output import OUTPUT_COLUMNS, format_row  # noqa: E402
from readers import (  # noqa: E402
    load_claims,
    load_evidence_requirements,
    load_user_history,
    parse_image_paths,
)
from validator import validate  # noqa: E402
from vision import create_genai_client, is_rate_limit_error, run_vision_pass  # noqa: E402


def _fallback_extraction(error: Exception) -> dict[str, Any]:
    message = str(error).strip() or error.__class__.__name__
    return {
        "valid_image": False,
        "issue_type": "unknown",
        "object_part": "unknown",
        "severity": "unknown",
        "supporting_image_ids": [],
        "visual_claim_status": "not_enough_information",
        "claim_status_justification": (
            f"Unhandled error during claim processing: {message}"
        ),
        "risk_flags": ["damage_not_visible"],
    }


def _write_predictions(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
    dataframe.to_csv(output_path, index=False, quoting=1)


def _resolve_paths(mode: str) -> tuple[Path, str]:
    if mode == "sample":
        return CODE_DIR / "evaluation" / "sample_predictions.csv", "sample_claims.csv"
    if mode == "test":
        return REPO_ROOT / "output.csv", "claims.csv"
    raise ValueError("Mode must be 'sample' or 'test'")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"sample", "test"}:
        print("Usage: python main.py <sample|test>", file=sys.stderr)
        sys.exit(1)

    mode = sys.argv[1]
    output_path, claims_filename = _resolve_paths(mode)

    claims_df = load_claims(claims_filename)
    history_df = load_user_history()
    requirements_df = load_evidence_requirements()
    client = create_genai_client()

    total_rows = len(claims_df)
    output_rows: list[dict[str, str]] = []
    error_count = 0
    started_at = time.perf_counter()

    for row_number, (_, claim_row) in enumerate(claims_df.iterrows(), start=1):
        user_id = str(claim_row["user_id"])
        
        output_row = None
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                extraction = run_vision_pass(
                    str(claim_row["image_paths"]),
                    str(claim_row["user_claim"]),
                    user_id,
                    str(claim_row["claim_object"]),
                    requirements_df,
                    client,
                )
                history_flags = get_history_risk_flags(user_id, history_df)
                validated_result, _fired_rules = validate(
                    extraction,
                    history_flags,
                    claim_object=str(claim_row["claim_object"]),
                    requirements_df=requirements_df,
                    num_images=len(parse_image_paths(str(claim_row["image_paths"]), user_id)),
                )
                output_row = format_row(
                    claim_row,
                    extraction,
                    history_flags,
                    validated_result,
                )
                break  # Success, break the retry loop
                
            except Exception as exc:
                if is_rate_limit_error(exc) and attempt < max_attempts - 1:
                    print(
                        f"  [Rate Limit] 429/Exhausted on user_id={user_id}. "
                        f"Sleeping 65s (Attempt {attempt + 1}/{max_attempts})...", 
                        file=sys.stderr
                    )
                    time.sleep(65)
                    continue  # Wait and try the same claim again
                
                # Hard failure or retries exhausted
                error_count += 1
                print(f"ERROR row={row_number} user_id={user_id}: {exc}", file=sys.stderr)
                traceback.print_exc()
                
                history_flags = get_history_risk_flags(user_id, history_df)
                fallback_extraction = _fallback_extraction(exc)
                validated_result, _fired_rules = validate(
                    fallback_extraction,
                    history_flags,
                    claim_object=str(claim_row["claim_object"]),
                    requirements_df=requirements_df,
                    num_images=len(parse_image_paths(str(claim_row["image_paths"]), user_id)),
                )
                output_row = format_row(
                    claim_row,
                    fallback_extraction,
                    history_flags,
                    validated_result,
                )
                break  # Break retry loop and proceed to next claim

        output_rows.append(output_row)
        print(
            f"ROW {len(output_rows)}/{total_rows} user_id={user_id} "
            f"claim_status={output_row['claim_status']}"
        )

    _write_predictions(output_rows, output_path)
    elapsed_seconds = time.perf_counter() - started_at

    print(f"Wrote {len(output_rows)} rows to {output_path}")
    print(f"Total rows: {total_rows}")
    print(f"Error count: {error_count}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()