# Evaluation Report — Multi-Modal Damage Claim Verification

## 1. Evaluation setup

- **Dev/sample set:** `dataset/sample_claims.csv` — 20 labeled rows, used for prompt and
  validator iteration.
- **Test set:** `dataset/claims.csv` — 44 unlabeled rows, final predictions in `output.csv`.
- **Comparator:** `code/evaluation/main.py` compares `evaluation/sample_predictions.csv`
  against `dataset/sample_claims.csv` field-by-field, normalizing list fields
  (`risk_flags`, `supporting_image_ids` — sorted, deduped, case-insensitive `none`
  handling) and boolean fields before comparing.

**Note on the comparator's "exact-match rate":** two of the fourteen output columns
(`evidence_standard_met_reason`, `claim_status_justification`) are free-text justifications.
Exact string comparison against hand-written ground-truth phrasing is not a meaningful
signal for these fields — an LLM-generated justification can be fully correct in substance
while never matching the labeler's exact wording. We treat the **structured/categorical
fields** as the primary accuracy signal and the free-text fields as qualitative only.

## 2. Strategy comparison (two configurations evaluated on the sample set)

| | Baseline | After fix |
|---|---|---|
| **Change** | Original validator: `manual_review_required` only appended when `claim_status=supported` with no supporting images | Added a deterministic rule: also append `manual_review_required` when `user_history_risk` is present in history flags, or `wrong_object`/`claim_mismatch` is present in vision risk flags |
| `risk_flags` accuracy | 8/20 (40%) | 11/20 (55%) |

This correlation was verified directly against ground truth before implementing: every
sample row with `user_history_risk` (6/6) and every row with `wrong_object`/`claim_mismatch`
(4/4) in the expected output also carried `manual_review_required`. The fix is purely
deterministic (no extra model calls), implemented in `validator.py`.

We also observed ~1 row of run-to-run variance across the two `python main.py sample`
calls at `temperature=0` (several unrelated fields — `issue_type`, `valid_image`,
`severity` — shifted by exactly one row between runs). This is consistent with known
non-determinism in hosted LLM APIs even at temperature 0, and is a caveat on any single
before/after sample comparison at this scale (20 rows ⇒ ±1 row ≈ ±5pp noise floor).

## 3. Sample-set accuracy (final, structured fields only)

From `python evaluation/main.py` (2026-06-20, final `sample_predictions.csv`):

| Field | Accuracy |
|---|---|
| `valid_image` | 18/20 (90.0%) |
| `supporting_image_ids` | 17/20 (85.0%) |
| `evidence_standard_met` | 14/20 (70.0%) |
| `object_part` | 16/20 (80.0%) |
| `claim_status` | 12/20 (60.0%) |
| `risk_flags` | 11/20 (55.0%) |
| `issue_type` | 9/20 (45.0%) |
| `severity` | 7/20 (35.0%) |

Free-text fields (`evidence_standard_met_reason`, `claim_status_justification`) scored 0/20
by exact match — expected, per §1. Overall exact-row match (all 14 prediction columns):
0/20 (0.0%).

## 4. Known limitations (not fixed due to time constraints)

- **No enum-clamping layer.** `VisionExtraction` (Pydantic model in `vision.py`) types
  `issue_type`, `object_part`, `severity`, `risk_flags` as plain `str`/`list[str]`, not
  constrained to the allowed-value lists in `config.py`. Nothing downstream
  (`validator.py`, `output.py`) checks membership either — `output.py` only lowercases
  three fields. In practice, all 44 test-set predictions landed inside the allowed enum
  lists (verified by direct check), but this is not enforced by the code and is a gap
  worth closing with more time.
- **`REQ_CAR_IDENTITY_OR_SIDE` is defined but unused.** `evidence_requirements.csv` has
  this requirement (min. 2 supporting images, for claims depending on vehicle
  identity/side), but `evidence.resolve_requirements()` never includes it in any matched
  rule set. Cross-image vehicle-identity mismatches are currently caught only by the
  vision prompt (`wrong_object`/`claim_mismatch`), with no deterministic backstop.
- **`wrong_object`/`claim_mismatch` over-triggering.** On the sample set, 3/20 rows
  received these flags when ground truth expected none. On the full test set, 17 of 21
  `not_enough_information` predictions (81%) are driven by these same flags — meaning
  the system is likely conservative/over-cautious about cross-image object consistency.
  This pushes more test-set rows into `not_enough_information` than may be warranted,
  but errs toward flagging for human review rather than wrongly auto-approving, which we
  consider an acceptable trade-off for an insurance-evidence-review context.
- **`check_evidence()` (in `evidence.py`) and the `requirements_df` parameter of
  `run_vision_pass()` are dead code** — defined/accepted but not used in the live
  pipeline. Left in place rather than removed, to avoid late-stage refactor risk.

## 5. Operational analysis

### Model calls
- **1 Gemini call per claim row** (single multi-image call per claim; no second LLM
  pass — history and evidence-requirement logic are pure CSV/code, zero LLM calls).
- Sample run: 20 calls. Test run: 44 calls. **Total: 64 model calls** for this submission
  (plus any earlier dev/debug runs not counted here).

### Images processed
- Test set (`claims.csv`): **82 images** across 44 rows (1–3 images per row; sum of
  semicolon-separated `image_paths` entries). Average 1.86 images/row.
- Sample set: **29 images** across 20 rows (1–2 images per row).
- Max images per claim is capped at 5 (`MAX_IMAGES` in `config.py`); excess images are
  dropped with a logged warning. No row in either set hit this cap.

### Token usage (approximate)
- **Not captured from API response metadata** — `main.py` / `vision.py` do not log
  usage fields. Estimate below uses char/4 for text and **258 tokens/image** (Gemini
  low-res image tile heuristic):
  - Sample run (20 calls): ~23,580 input + ~4,000 output tokens
  - Test run (44 calls): ~55,862 input + ~8,800 output tokens
  - **Submission total (64 calls): ~79,400 input + ~12,800 output ≈ 92,200 tokens**
  - Per call (blended): ~1,240 input + ~200 output structured JSON

### Cost estimate
- Model: `gemini-3.1-flash-lite` (`GEMINI_MODEL` default in `config.py`).
- Paid-tier list price (Gemini 3.1 Flash-Lite, June 2026): **$0.25 / 1M input**,
  **$1.50 / 1M output** ([Google pricing](https://ai.google.dev/gemini-api/docs/pricing)).
- Estimated submission cost (64 calls, token estimate above):
  - Input: 79,400 × $0.25/1M ≈ **$0.02**
  - Output: 12,800 × $1.50/1M ≈ **$0.02**
  - **Total ≈ $0.04** (paid tier; free-tier quota may reduce actual spend to $0)

### Latency / runtime
- **Test run** (`python main.py test`): **259.41 s** wall-clock (`main.py` log)
- **Sample run** (`python main.py sample`): **110.16 s** wall-clock
- **Combined submission runs:** **369.57 s** (~6.2 min) for 64 model calls
- Per-call latency (wall-clock, sequential):
  - Test: 259.41 / 44 ≈ **5.9 s/row**
  - Sample: 110.16 / 20 ≈ **5.5 s/row**
  - Blended: 369.57 / 64 ≈ **5.8 s/row**
- One transient **503 UNAVAILABLE** (high demand) on test row 28 (`user_041`); vision
  fell back to `not_enough_information` internally — not counted in `Error count`.

### Rate limits, retries, batching
- **No batching** — one synchronous call per row, sequential.
- **Retry strategy:** up to 3 attempts per row on rate-limit/quota errors
  (`is_rate_limit_error()` in `vision.py`), flat 65-second sleep between attempts
  (`main.py`). Non-rate-limit failures fall back immediately to a
  `not_enough_information` / `damage_not_visible` row rather than retrying.
- **No caching implemented.** Each run re-calls the model for every row; re-running the
  same dataset twice doubles API cost. A disk cache keyed on
  `hash(images + user_claim + claim_object)` was planned but not implemented in this
  version — noted as a clear next improvement for cost control during iteration.
- **Error count on final test run:** **0** (`main.py` printed `Error count: 0`; sample
  run also 0). One 503 on row 28 was handled inside `vision.py` via fallback extraction.

## 6. Summary

The system uses a single deterministic-temperature VLM call per claim plus a fully
rule-based (zero-LLM) layer for evidence-requirement checking, cross-field consistency
validation, and history-risk merging — keeping cost and latency to one model call per
row. The main accuracy gaps on the sample set are concentrated in `severity`
calibration (systematic over-estimation toward "high") and `risk_flags`
(`manual_review_required` correlation, since fixed; cross-image mismatch
over-triggering, not yet fixed). Both are prompt-level issues in the vision pass rather
than architectural ones, and are documented as known limitations above rather than
silently left unmentioned.
