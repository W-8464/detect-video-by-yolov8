#!/usr/bin/env python3
"""
Regression test suite for DynamicSOPBuilder.

Compares the current inference output against saved "golden" snapshots
to detect unintended side‑effects when adding new actions or tuning parameters.

Usage
-----
# First run — generate golden snapshots for all available datasets:
    python tests/test_regression_sop.py --update

# Subsequent runs — verify nothing changed:
    python tests/test_regression_sop.py

# Run with pytest:
    pytest tests/test_regression_sop.py -v

# Verify one dataset only:
    pytest tests/test_regression_sop.py -k xb10_2_8_cut -v
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dynamic_sop_builder import DynamicSOPBuilder, load_yaml

# ── Directory layout ────────────────────────────────────────────────────
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden_sop"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

# ── Test registry ───────────────────────────────────────────────────────
# Each entry maps a short name to:
#   detections  — path to the *_detections.jsonl
#   templates   — list of template yaml filenames
#   global_cfg  — path to the global config yaml
#
# Add new datasets here when you want them covered by regression tests.

DEFAULT_TEMPLATES = [
    "put_board.yaml",
    "take_only.yaml",
    "take_gasket.yaml",
    "apply_gasket_to_shielding.yaml",
    "attach_gasket_payload_to_board.yaml",
    "attach_screw.yaml",
    "attach_only.yaml",
    "return_board.yaml",
]

# Auto-discover datasets by looking for *_detections.jsonl files
def _discover_datasets() -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    for jsonl in sorted(PROJECT_ROOT.glob("*_detections.jsonl")):
        name = jsonl.name.replace("_detections.jsonl", "")
        # Skip world detections or other special files
        if "world" in name:
            continue
        registry[name] = {
            "detections": str(jsonl),
            "templates": [str(PROJECT_ROOT / t) for t in DEFAULT_TEMPLATES],
            "global_cfg": str(PROJECT_ROOT / "sop_global_shared.yaml"),
        }
    return registry


DATASET_REGISTRY = _discover_datasets()


# ── Golden snapshot I/O ─────────────────────────────────────────────────

def _golden_path(dataset_name: str) -> Path:
    return GOLDEN_DIR / f"{dataset_name}.golden.json"


def _extract_action_summary(inferred: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract only the fields we care about for regression comparison.
    Keeps things stable even if cosmetic output fields change.
    """
    matches = inferred.get("_inference_meta", {}).get("matches", [])
    summary: List[Dict[str, Any]] = []
    for m in matches:
        summary.append({
            "action_id": m["action_id"],
            "source_template": Path(m["source_template"]).name,
            "start_frame": m["start_frame"],
            "end_frame": m["end_frame"],
            "confidence": round(float(m.get("confidence", 0)), 3),
        })
    return summary


def _run_inference(dataset_name: str) -> Dict[str, Any]:
    """Run DynamicSOPBuilder for a dataset and return the raw inferred dict."""
    cfg = DATASET_REGISTRY[dataset_name]
    template_paths = [Path(p) for p in cfg["templates"]]
    # Filter to only existing templates (e.g. attach_screw.yaml may not exist in old checkouts)
    template_paths = [p for p in template_paths if p.exists()]
    global_cfg = load_yaml(Path(cfg["global_cfg"])).get("global", {})
    builder = DynamicSOPBuilder(
        template_paths=template_paths,
        global_cfg=global_cfg,
    )
    return builder.infer_from_detections(Path(cfg["detections"]))


def save_golden(dataset_name: str, summary: List[Dict[str, Any]]) -> Path:
    """Persist golden snapshot."""
    path = _golden_path(dataset_name)
    data = {
        "dataset": dataset_name,
        "action_count": len(summary),
        "actions": summary,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_golden(dataset_name: str) -> Optional[List[Dict[str, Any]]]:
    """Load previously saved golden snapshot, or None."""
    path = _golden_path(dataset_name)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("actions", [])


# ── Comparison helpers ──────────────────────────────────────────────────

FRAME_TOLERANCE = 5  # allow ±5 frames of drift without failing


def compare_summaries(
    golden: List[Dict[str, Any]],
    actual: List[Dict[str, Any]],
    frame_tol: int = FRAME_TOLERANCE,
) -> List[str]:
    """
    Compare two action summaries and return a list of human-readable
    differences. Empty list = no regression.
    """
    diffs: List[str] = []

    if len(golden) != len(actual):
        diffs.append(
            f"ACTION COUNT changed: golden={len(golden)}, actual={len(actual)}"
        )

    # Compare actions in order up to shorter list
    for i in range(max(len(golden), len(actual))):
        g = golden[i] if i < len(golden) else None
        a = actual[i] if i < len(actual) else None

        prefix = f"[action #{i+1}]"
        if g is None:
            diffs.append(f"{prefix} ADDED: {a['action_id']}")
            continue
        if a is None:
            diffs.append(f"{prefix} REMOVED: {g['action_id']}")
            continue

        if g["action_id"] != a["action_id"]:
            diffs.append(
                f"{prefix} ID changed: '{g['action_id']}' → '{a['action_id']}'"
            )
        if g["source_template"] != a["source_template"]:
            diffs.append(
                f"{prefix} template changed: '{g['source_template']}' → '{a['source_template']}'"
            )
        sf_drift = abs(g["start_frame"] - a["start_frame"])
        ef_drift = abs(g["end_frame"] - a["end_frame"])
        if sf_drift > frame_tol:
            diffs.append(
                f"{prefix} {g['action_id']} start_frame drifted: "
                f"{g['start_frame']} → {a['start_frame']} (Δ={sf_drift})"
            )
        if ef_drift > frame_tol:
            diffs.append(
                f"{prefix} {g['action_id']} end_frame drifted: "
                f"{g['end_frame']} → {a['end_frame']} (Δ={ef_drift})"
            )

    return diffs


# ── Pytest parametrised tests ──────────────────────────────────────────

@pytest.fixture(scope="module", params=sorted(DATASET_REGISTRY.keys()))
def dataset_name(request):
    return request.param


def test_regression(dataset_name: str):
    """
    For each registered dataset:
    1. Load its golden snapshot.
    2. Re-run inference.
    3. Compare and fail if differences are found.

    If no golden snapshot exists, the test is SKIPPED (run --update first).
    """
    golden = load_golden(dataset_name)
    if golden is None:
        pytest.skip(
            f"No golden snapshot for '{dataset_name}'. "
            f"Run: python tests/test_regression_sop.py --update"
        )

    inferred = _run_inference(dataset_name)
    actual = _extract_action_summary(inferred)
    diffs = compare_summaries(golden, actual)

    if diffs:
        msg = f"\n🔴 REGRESSION in {dataset_name}:\n" + "\n".join(f"  • {d}" for d in diffs)
        # Also show side-by-side
        msg += "\n\n  Golden actions:"
        for g in golden:
            msg += f"\n    {g['action_id']:40s} [{g['start_frame']:5d} – {g['end_frame']:5d}]"
        msg += "\n\n  Actual actions:"
        for a in actual:
            msg += f"\n    {a['action_id']:40s} [{a['start_frame']:5d} – {a['end_frame']:5d}]"
        pytest.fail(msg)


# ── CLI: update golden snapshots ────────────────────────────────────────

def update_all_goldens(datasets: Optional[List[str]] = None):
    """Run inference on all (or selected) datasets and save golden snapshots."""
    targets = datasets or sorted(DATASET_REGISTRY.keys())
    for name in targets:
        if name not in DATASET_REGISTRY:
            print(f"⚠️  Unknown dataset: {name}, skipping")
            continue
        print(f"⏳ Running inference for {name}…", end=" ", flush=True)
        try:
            inferred = _run_inference(name)
            summary = _extract_action_summary(inferred)
            path = save_golden(name, summary)
            action_ids = [a["action_id"] for a in summary]
            print(f"✅ {len(summary)} actions → {path.name}")
            for aid in action_ids:
                print(f"     {aid}")
        except Exception as e:
            print(f"❌ Error: {e}")


def verify_all(datasets: Optional[List[str]] = None):
    """Quick CLI verification without pytest."""
    targets = datasets or sorted(DATASET_REGISTRY.keys())
    all_pass = True
    for name in targets:
        if name not in DATASET_REGISTRY:
            continue
        golden = load_golden(name)
        if golden is None:
            print(f"⏭️  {name}: no golden snapshot (run --update first)")
            continue
        print(f"⏳ Verifying {name}…", end=" ", flush=True)
        try:
            inferred = _run_inference(name)
            actual = _extract_action_summary(inferred)
            diffs = compare_summaries(golden, actual)
            if diffs:
                print(f"🔴 FAIL")
                for d in diffs:
                    print(f"     {d}")
                all_pass = False
            else:
                print(f"✅ PASS ({len(actual)} actions)")
        except Exception as e:
            print(f"❌ Error: {e}")
            all_pass = False
    return all_pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SOP Builder Regression Test")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Re-run inference on all datasets and save as golden snapshots.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Only process these datasets (default: all discovered).",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=FRAME_TOLERANCE,
        help=f"Frame tolerance for start/end comparison (default: {FRAME_TOLERANCE}).",
    )
    args = parser.parse_args()

    FRAME_TOLERANCE = args.tolerance

    if args.update:
        update_all_goldens(args.datasets)
    else:
        ok = verify_all(args.datasets)
        sys.exit(0 if ok else 1)
