"""Audit LLM response completeness: expected-key left join.

For each cached LLM prediction file, compare expected records (one per
ground-truth decision) against parsed predictions and report:
  - participants present / expected
  - decision-level coverage
  - missing (expected decision with no parsed prediction)
  - invalid label (parsed label not grant/deny-mappable)
  - duplicate (same query id predicted more than once)
Writes results/table_llm_response_audit.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

VALID = {
    "yes, always share", "yes, share", "yes", "grant", "allow",
    "no, never share", "no, do not share", "no", "deny", "refuse",
}


def normalize(label):
    if not isinstance(label, str):
        return None
    return label.strip().lower()


def audit_file(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    n_participants = len(d)
    expected = parsed = missing = invalid = duplicate = 0
    for pid, rec in d.items():
        gt = rec.get("ground_truth", [])
        preds = rec.get("predictions", [])
        expected += len(gt)
        parsed += len(preds)
        seen_ids = set()
        for p in preds:
            qid = p.get("id")
            if qid in seen_ids:
                duplicate += 1
            seen_ids.add(qid)
            perms = p.get("permission") or {}
            if isinstance(perms, dict):
                for _k, v in perms.items():
                    lab = normalize((v or {}).get("label") if isinstance(v, dict) else None)
                    if lab is None or lab not in VALID:
                        invalid += 1
        n_missing = max(0, len(gt) - len(preds))
        missing += n_missing
    return {
        "file": path.name,
        "participants": n_participants,
        "expected_decisions": expected,
        "parsed_predictions": parsed,
        "missing_decisions": missing,
        "invalid_labels": invalid,
        "duplicate_predictions": duplicate,
        "decision_coverage_pct": round(100 * parsed / expected, 2) if expected else None,
    }


def main():
    files = [
        "ic_only_predictions.json",
        "ic_cf_predictions.json",
        "ic_only_gpt4omini_predictions.json",
        "ic_only_predictions_o4mini.json",
        "ic_only_promptvar_predictions.json",
    ]
    out = {}
    for f in files:
        p = RESULTS / f
        if p.exists():
            out[f.replace(".json", "")] = audit_file(p)
    dest = RESULTS / "table_llm_response_audit.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
