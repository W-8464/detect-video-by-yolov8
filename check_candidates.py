"""
Check frames around 153-155 against take_only.yaml conditions using dynamic_sop_builder.py logic.
"""
import json
import sys
from pathlib import Path

import numpy as np

from segment_actions_refactored import Detection, evaluate_condition, aabb, iou_aabb, contain_ratio_aabb

DETECTIONS_PATH = Path("/home/admin/my_yolo_project/xb10_2_8_detections.jsonl")

# Class groups from sop_global_shared.yaml
CLASS_GROUPS = {
    "Component": ["PCIe cable", "shielding", "gasket"],
    "Tool": ["tweezers"],
    "Target": ["board", "bracket"],
    "Container": ["tray"],
}


def load_frames(path: Path):
    by_frame = {}
    max_frame = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        fi = int(row.get("frame", -1))
        if fi < 0:
            continue
        max_frame = max(max_frame, fi)
        dets = []
        for d in row.get("detections", []):
            cls_name = str(d.get("class", d.get("cls", ""))).strip()
            if not cls_name:
                continue
            conf = float(d.get("conf", 0.0))
            poly = d.get("poly")
            if not poly:
                continue
            dets.append(Detection(cls=cls_name, conf=conf, poly=np.array(poly), track_id=d.get("track_id", None)))
        by_frame[fi] = dets

    frames = [by_frame.get(i, []) for i in range(1, max_frame + 1)]
    return frames


def check_phase(frames, frame_idx, conditions, history_len=30):
    """Check if a single frame satisfies all phase conditions."""
    history = frames[max(0, frame_idx - history_len):frame_idx]
    return all(evaluate_condition(frames[frame_idx], c, CLASS_GROUPS, history) for c in conditions)


def main():
    frames = load_frames(DETECTIONS_PATH)
    print(f"Loaded {len(frames)} frames")

    # Phase 0 conditions from take_only.yaml
    phase0_conds = [
        {"type": "contain", "subject": "hand", "target": "Container", "min_thr": 0.35},
        {"type": "not_contain", "subject": "hand", "target": "Target", "max_thr": 0.30},
        {"type": "stationary", "subject": "Target", "history_len": 15, "min_iou": 0.85},
        {"type": "grasp", "subject": "hand", "min_thr": 0.4},
        {"type": "or", "sub_conditions": [
            {"type": "iou", "subject": "hand", "target": "Component", "min_thr": 0.005},
            {"type": "history_exists", "subject": "Component", "history_len": 8, "min_frames": 3},
        ]},
    ]

    # Phase 1 conditions
    phase1_conds = [
        {"type": "or", "sub_conditions": [
            {"type": "not_contain", "subject": "hand", "target": "Container", "max_thr": 0.30},
            {"type": "contain", "subject": "hand", "target": "Target", "min_thr": 0.30},
        ]},
    ]

    # Scan frames around 153-155 (wider range: 100-250)
    start_scan = max(0, 100)
    end_scan = min(len(frames), 250)

    print(f"\n=== Scanning frames {start_scan+1} to {end_scan} ===")
    print(f"{'Frame':>6} | {'Phase0':>7} | {'Phase1':>7} | Hands present | Components present")
    print("-" * 90)

    phase0_hits = []
    for fi in range(start_scan, end_scan):
        p0 = check_phase(frames, fi, phase0_conds)
        p1 = check_phase(frames, fi, phase1_conds)

        frame_dets = frames[fi]
        hands = [d for d in frame_dets if d.cls == "hand"]
        comps = [d for d in frame_dets if d.cls in CLASS_GROUPS["Component"]]
        trays = [d for d in frame_dets if d.cls == "tray"]
        targets = [d for d in frame_dets if d.cls in CLASS_GROUPS["Target"]]

        hand_str = ", ".join(f"h(tid={h.track_id}, conf={h.conf:.2f})" for h in hands)
        comp_str = ", ".join(f"{c.cls}(tid={c.track_id}, conf={c.conf:.2f})" for c in comps)

        if p0 or p1:
            marker = " <<<" if p0 else ""
            print(f"{fi+1:>6} | {'YES':>7} | {'YES' if p1 else 'NO':>7} | {hand_str} | {comp_str}{marker}")
            if p0:
                phase0_hits.append(fi)
        elif fi in [152, 153, 154]:  # always show frames 153-155 (0-indexed: 152-154)
            print(f"{fi+1:>6} | {'NO':>7} | {'NO':>7} | {hand_str} | {comp_str}")

    print(f"\n=== Phase 0 hits: {len(phase0_hits)} frames ===")
    if phase0_hits:
        # Group consecutive hits
        groups = []
        cur = [phase0_hits[0]]
        for h in phase0_hits[1:]:
            if h == cur[-1] + 1:
                cur.append(h)
            else:
                groups.append(cur)
                cur = [h]
        groups.append(cur)
        for g in groups:
            print(f"  Frames {g[0]+1}-{g[-1]+1} ({len(g)} frames)")

    # Now check for full candidate matches using the builder's try_match logic
    print("\n=== Checking for full candidate matches (phase0 hold=4, phase1 hold=3) ===")
    from dynamic_sop_builder import ActionTemplate

    template = ActionTemplate(Path("/home/admin/my_yolo_project/take_only.yaml"))
    for start in range(0, len(frames), 4):  # scan_stride=4
        start_rel, end_rel, conf, timer_rel, _ = template.try_match(
            frames[start:start + 300], CLASS_GROUPS
        )
        if end_rel > 0 and conf > 0:
            end_abs = min(len(frames), start + end_rel)
            semantic_start = start + start_rel + 1 if start_rel >= 0 else start + 1
            timer_start = start + timer_rel + 1 if timer_rel >= 0 else semantic_start
            detected = template.get_detected_component()
            # Only show candidates near frame 153-155
            if abs(semantic_start - 153) < 60:
                print(f"  CANDIDATE: start={start+1}, semantic_start={semantic_start}, "
                      f"timer_start={timer_start}, end={end_abs}, conf={conf:.4f}, "
                      f"component={detected}")


if __name__ == "__main__":
    main()
