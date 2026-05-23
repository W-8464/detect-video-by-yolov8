import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MAKE_CLIPS = False
RENDER_TIMER_VIDEO = False

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing PyYAML. Install with: pip install pyyaml") from exc


@dataclass
class Detection:
    cls: str
    conf: float
    poly: np.ndarray  # shape (N, 2)
    track_id: Optional[int]


def polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def aabb(poly: np.ndarray) -> Tuple[float, float, float, float]:
    return float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())


def iou_aabb(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + ab - inter
    return float(inter / den) if den > 0 else 0.0


def inter_area_aabb(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def best_single_tray_for_hand(
    hand: Detection,
    hand_box: Tuple[float, float, float, float],
    trays: List[Detection],
    min_iou: float,
) -> Tuple[Optional[Detection], float]:
    """When hand overlaps multiple trays, keep only the tray with largest overlap area."""
    best_tray: Optional[Detection] = None
    best_iou = 0.0
    best_inter = 0.0
    for t in trays:
        t_box = aabb(t.poly)
        if not center_in_aabb(hand.poly, t_box):
            continue
        ov = iou_aabb(hand_box, t_box)
        if ov < min_iou:
            continue
        inter = inter_area_aabb(hand_box, t_box)
        if best_tray is None or inter > best_inter or (inter == best_inter and ov > best_iou):
            best_tray = t
            best_iou = ov
            best_inter = inter
    return best_tray, best_iou


def contain_ratio_aabb(inner: Tuple[float, float, float, float], outer: Tuple[float, float, float, float]) -> float:
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    inner_area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    if inner_area <= 0:
        return 0.0
    return float(inter / inner_area)


def center_in_aabb(poly: np.ndarray, box: Tuple[float, float, float, float]) -> bool:
    c = poly.mean(axis=0)
    return bool(box[0] <= c[0] <= box[2] and box[1] <= c[1] <= box[3])


def top_point_in_aabb(poly: np.ndarray, box: Tuple[float, float, float, float]) -> bool:
    idx = int(np.argmin(poly[:, 1]))
    p = poly[idx]
    return bool(box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3])


def cover_ratio(obj_poly: np.ndarray, zone_poly: np.ndarray) -> float:
    # Approximation: AABB IoU (fast, dependency-free). Replace with polygon intersection if needed.
    return iou_aabb(aabb(obj_poly), aabb(zone_poly))


def read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def to_abs_poly(norm_poly: List[List[float]], w: int, h: int) -> np.ndarray:
    p = np.array(norm_poly, dtype=np.float32)
    p[:, 0] *= float(w)
    p[:, 1] *= float(h)
    return p


def load_detections_jsonl(path: Path, min_conf: float) -> Dict[int, List[Detection]]:
    """
    Supported JSONL line formats:
    1) {"frame": 123, "detections":[{"class":"board","conf":0.9,"track_id":1,"poly":[[x,y]...]}]}
    2) {"frame": 123, "class":"board","conf":0.9,"track_id":1,"poly":[[x,y]...]}
    """
    by_frame: Dict[int, List[Detection]] = {}
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        frame = int(row.get("frame", -1))
        if frame < 0:
            continue
        items = row.get("detections", None)
        if items is None:
            items = [row]
        out = by_frame.setdefault(frame, [])
        for d in items:
            cls = str(d.get("class", d.get("cls", "")))
            conf = float(d.get("conf", 0.0))
            if not cls or conf < min_conf:
                continue
            poly = d.get("poly", None)
            if not poly:
                # fallback from center/box if available
                cx = d.get("cx", None)
                cy = d.get("cy", None)
                bw = d.get("w", None)
                bh = d.get("h", None)
                if None not in (cx, cy, bw, bh):
                    x1 = float(cx) - float(bw) / 2.0
                    y1 = float(cy) - float(bh) / 2.0
                    x2 = float(cx) + float(bw) / 2.0
                    y2 = float(cy) + float(bh) / 2.0
                    poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                else:
                    continue
            out.append(
                Detection(
                    cls=cls,
                    conf=conf,
                    poly=np.array(poly, dtype=np.float32),
                    track_id=d.get("track_id", None),
                )
            )
    return by_frame


def best_zone_poly(frame_dets: List[Detection], source_cfg: dict, static_zones: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    stype = source_cfg.get("type", "none")
    if stype == "none":
        return None
    if stype == "static_zone":
        return static_zones.get(str(source_cfg.get("zone_name", "")), None)
    if stype == "object_class":
        c = str(source_cfg.get("class_name", ""))
        cand = [d for d in frame_dets if d.cls == c]
        if not cand:
            return None
        cand.sort(key=lambda d: d.conf, reverse=True)
        return cand[0].poly
    return None


def action_score(
    frame_dets: List[Detection],
    action: dict,
    static_zones: Dict[str, np.ndarray],
    ema_prev: float,
    alpha: float,
) -> Tuple[float, float, dict]:
    obj_cls = str(action["object_class"])
    obj_cands = [d for d in frame_dets if d.cls == obj_cls]
    if not obj_cands:
        ema = (1 - alpha) * ema_prev
        return 0.0, ema, {"space": 0.0, "time": ema, "manip": 0.0}
    obj_cands.sort(key=lambda d: d.conf, reverse=True)
    obj = obj_cands[0]

    source_poly = best_zone_poly(frame_dets, action.get("source", {"type": "none"}), static_zones)
    target_poly = best_zone_poly(frame_dets, action.get("target", {"type": "none"}), static_zones)

    src_cov = cover_ratio(obj.poly, source_poly) if source_poly is not None else 0.0
    tgt_cov = cover_ratio(obj.poly, target_poly) if target_poly is not None else 0.0

    # Space score depends on action type:
    # - source -> target: prefer object moving to target while leaving source
    # - source only: pick-like action, prefer leaving source
    # - target only: place-like action, prefer entering target
    if source_poly is not None and target_poly is not None:
        space = float(max(0.0, tgt_cov - 0.5 * src_cov))
    elif source_poly is not None and target_poly is None:
        space = float(max(0.0, 1.0 - src_cov))
    elif source_poly is None and target_poly is not None:
        space = float(max(0.0, tgt_cov))
    else:
        space = 0.0
    space = min(1.0, max(0.0, space))

    ema = alpha * space + (1.0 - alpha) * ema_prev
    time_score = float(min(1.0, max(0.0, ema)))

    hand_contact = 0.0
    tool_contact = 0.0
    hand_dets = [d for d in frame_dets if d.cls == "hand"]
    tw_dets = [d for d in frame_dets if d.cls == "tweezers"]
    if hand_dets:
        hand_contact = max(iou_aabb(aabb(obj.poly), aabb(h.poly)) for h in hand_dets)
    if tw_dets:
        tool_contact = max(iou_aabb(aabb(obj.poly), aabb(t.poly)) for t in tw_dets)

    manip_cfg = action.get("manipulation", {})
    req_hand = bool(manip_cfg.get("require_hand_contact", False))
    req_tool = bool(manip_cfg.get("require_tool_contact", False))
    hand_ok = 1.0 if (not req_hand or hand_contact > 0.02) else 0.0
    tool_ok = 1.0 if (not req_tool or tool_contact > 0.02) else 0.0
    manip = 0.5 * hand_ok + 0.5 * tool_ok
    if not req_hand and not req_tool:
        manip = 1.0

    w = action.get("weights", {"space": 0.5, "time": 0.25, "manipulation": 0.25})
    score = float(w.get("space", 0.5) * space + w.get("time", 0.25) * time_score + w.get("manipulation", 0.25) * manip)
    score = min(1.0, max(0.0, score))
    return score, ema, {"space": space, "time": time_score, "manip": manip}


def cut_clip(video_path: Path, out_path: Path, start_frame: int, end_frame: int) -> None:
    global MAKE_CLIPS
    if not MAKE_CLIPS:
        # When clips are disabled, avoid writing any mp4 files.
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    cur = start_frame
    while cur <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        cur += 1
    writer.release()
    cap.release()


def render_video_with_timer_overlay(
    video_path: Path,
    out_path: Path,
    fps: float,
    timeline_rows: List[dict],
) -> None:
    """Render full video (no slicing) with an on-screen timer table overlay.

    Requirements:
    - Show `action_id` + elapsed seconds while that action is active.
    - When an action ends, freeze its elapsed time.
    - Next action appears on the next line and starts its own timer.
    - Gaps not covered by any action show `idle` and do NOT show a timer.
    """
    if fps <= 0:
        fps = 30.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    if fps_in <= 0:
        fps_in = fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (w, h))
    if not writer.isOpened():
        cap.release()
        return

    # Build action segments from timeline and fill gaps with "idle".
    action_segs: List[Tuple[int, int, str]] = []
    for r in timeline_rows:
        act_id = str(r.get("action_id", ""))
        if not act_id or act_id == "Idle":
            continue
        if r.get("start_frame", None) is None or r.get("end_frame", None) is None:
            continue
        s = int(r.get("start_frame", -1))
        e = int(r.get("end_frame", -1))
        if s < 0 or e < s:
            continue
        action_segs.append((s, e, act_id))
    action_segs.sort(key=lambda x: x[0])

    segments: List[Tuple[int, int, str]] = []
    cursor = 0
    for s, e, act_id in action_segs:
        if s > cursor:
            segments.append((cursor, s - 1, "idle"))
        segments.append((s, e, act_id))
        cursor = e + 1
    if total_frames > 0 and cursor < total_frames:
        segments.append((cursor, total_frames - 1, "idle"))

    seg_idx = 0
    frame_idx = 0

    # Keep last N rows for readability; newest rows appear at the bottom.
    rows: List[dict] = []
    max_rows = 10

    # Overlay layout
    x0, y0 = 10, 10
    line_h = 22
    pad = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_th = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Resolve current segment for this frame.
        while seg_idx < len(segments) and frame_idx > segments[seg_idx][1]:
            seg_idx += 1
        cur = segments[seg_idx] if seg_idx < len(segments) else (frame_idx, frame_idx, "idle")
        cur_s, cur_e, cur_id = int(cur[0]), int(cur[1]), str(cur[2])

        # Start a new display row when entering a new segment id.
        last_id = str(rows[-1]["action_id"]) if rows else ""
        if not rows or last_id != cur_id:
            # Freeze previous row if it was an action.
            if rows and rows[-1]["action_id"] != "idle":
                prev = rows[-1]
                prev_end = int(prev["end_frame"])
                prev["frozen_seconds"] = (prev_end - int(prev["start_frame"]) + 1) / fps_in

            rows.append(
                {
                    "action_id": cur_id,
                    "start_frame": cur_s,
                    "end_frame": cur_e,
                    "frozen_seconds": None,
                }
            )
            if len(rows) > max_rows:
                rows = rows[-max_rows:]

        # Keep end_frame synced (mostly stable).
        rows[-1]["end_frame"] = cur_e

        # Build text lines.
        lines: List[str] = []
        for r in rows:
            aid = str(r["action_id"])
            if aid == "idle":
                lines.append("idle")
                continue
            frozen = r.get("frozen_seconds")
            if frozen is None:
                elapsed = (frame_idx - int(r["start_frame"]) + 1) / fps_in
            else:
                elapsed = float(frozen)
            lines.append(f"{aid}  {elapsed:.2f}s")

        # Background box sized to content.
        box_w = 0
        for txt in lines:
            (tw, _), _ = cv2.getTextSize(txt, font, font_scale, font_th)
            box_w = max(box_w, tw)
        box_w = min(w - 20, box_w + 2 * pad)
        box_h = min(h - 20, len(lines) * line_h + 2 * pad)
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), thickness=-1)

        y = y0 + pad + 16
        for txt in lines:
            cv2.putText(frame, txt, (x0 + pad, y), font, font_scale, (255, 255, 255), font_th, cv2.LINE_AA)
            y += line_h

        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()


def apply_neighbor_segment_fallback(
    timeline_rows: List[dict],
    actions: List[dict],
    neighbor_fb_cfg: dict,
    video_path: Path,
    clips_dir: Path,
    global_pre_b: int,
    global_post_b: int,
) -> None:
    """Insert a segment from prev.end+1 to next.start-1 when target action has no row."""
    if not neighbor_fb_cfg.get("enabled"):
        return
    target_id = str(neighbor_fb_cfg.get("action_id", ""))
    prev_id = str(neighbor_fb_cfg.get("prev_action_id", ""))
    next_id = str(neighbor_fb_cfg.get("next_action_id", ""))
    if not target_id or not prev_id or not next_id:
        return
    by_id = {str(r["action_id"]): r for r in timeline_rows if r.get("action_id")}
    if target_id in by_id:
        return
    prev_r = by_id.get(prev_id)
    next_r = by_id.get(next_id)
    if not prev_r or not next_r:
        return
    s = int(prev_r["end_frame"]) + 1
    e = int(next_r["start_frame"]) - 1
    if e < s:
        return
    order: Optional[int] = None
    desc = ""
    for i, a in enumerate(actions):
        if str(a.get("id", "")) == target_id:
            order = i + 1
            desc = str(a.get("description", target_id))
            break
    if order is None:
        return
    pre_b = int(neighbor_fb_cfg.get("pre_buffer_frames", global_pre_b))
    post_b = int(neighbor_fb_cfg.get("post_buffer_frames", global_post_b))
    clip_start = max(0, s - pre_b)
    clip_end = e + post_b
    clip_name = f"{order:02d}_{target_id}.mp4"
    cut_clip(video_path, clips_dir / clip_name, clip_start, clip_end)
    timeline_rows.append(
        {
            "order": order,
            "action_id": target_id,
            "description": desc + " [neighbor fallback]",
            "start_frame": s,
            "end_frame": e,
            "clip_start_frame": clip_start,
            "clip_end_frame": clip_end,
            "best_score": round(float(neighbor_fb_cfg.get("fallback_best_score", -1.0)), 4),
            "clip_path": str(clips_dir / clip_name),
            "segment_source": "neighbor_fallback",
        }
    )


def compute_heuristic_reliability_0_100(
    row: dict,
    dbg: Optional[dict],
    act_mode: str,
) -> float:
    """Heuristic 0–100 'tin cậy nội bộ' — không phải độ chính xác so ground truth.

    Kết hợp outcome / end_reason / neighbor-fallback / mode / best_score và optional debug from-tray.
    """
    act_id = str(row.get("action_id", ""))
    if act_id == "Idle":
        return 0.0

    outcome = str(row.get("outcome", ""))
    end_reason = str(row.get("end_reason", ""))
    seg_source = str(row.get("segment_source", ""))
    try:
        bs = float(row.get("best_score", 0) or 0)
    except (TypeError, ValueError):
        bs = 0.0

    s = 50.0

    if outcome == "completed":
        s += 24.0
    elif outcome == "pending_handoff":
        s -= 14.0
    elif outcome == "fallback_segment":
        s -= 20.0

    if seg_source == "neighbor_fallback":
        s -= 10.0

    if end_reason == "handoff_prev_end_to_next_start":
        s -= 18.0
    elif end_reason == "tray_return_streak_met":
        s += 12.0
    elif end_reason in ("release_consecutive_met", "release_timer_met"):
        s += 6.0
    elif "max_action_frames_cap" in end_reason:
        s -= 12.0
    elif "scan_exhausted" in end_reason or end_reason in ("phase2_no_end", "end_item_lost_after_grace"):
        s -= 12.0
    elif end_reason.endswith(":end") and "handoff" not in end_reason:
        s += 8.0

    if act_mode == "board_to_jig_v1":
        s += 16.0 * min(1.0, max(0.0, bs))
    elif act_mode == "board_absent_end_v1":
        s += 12.0 * min(1.0, max(0.0, bs))
    elif act_mode == "from_tray_pick_score_v1":
        s += min(8.0, max(0.0, bs) * 45.0)

    if dbg:
        pf = dbg.get("primary_failure")
        if pf and outcome != "completed":
            s -= 5.0
        pr = str(dbg.get("phase2_break_reason") or "")
        if pr == "end_item_lost_after_grace":
            s -= 10.0

    return float(max(0.0, min(100.0, round(s, 2))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Score-based action segmentation from detections + config.")
    ap.add_argument("--video", required=True, help="Input video path.")
    ap.add_argument("--detections", required=True, help="Detections jsonl path.")
    ap.add_argument("--config", default="actions_config.yaml", help="Action config yaml path.")
    ap.add_argument("--out-dir", default="runs/actions", help="Output directory.")
    ap.add_argument(
        "--make-clips",
        action="store_true",
        help="If set, cut and write clips (mp4) for each action; otherwise skip video slicing.",
    )
    ap.add_argument(
        "--render-timer-video",
        action="store_true",
        help="If set, render a full-length mp4 with an elapsed-time overlay (no slicing).",
    )
    args = ap.parse_args()

    global MAKE_CLIPS
    MAKE_CLIPS = bool(getattr(args, "make_clips", False))
    global RENDER_TIMER_VIDEO
    RENDER_TIMER_VIDEO = bool(getattr(args, "render_timer_video", False))

    video_path = Path(args.video).resolve()
    det_path = Path(args.detections).resolve()
    cfg_path = Path(args.config).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    if MAKE_CLIPS:
        clips_dir.mkdir(parents=True, exist_ok=True)

    cfg = read_config(cfg_path)
    g = cfg.get("global", {})
    min_conf = float(g.get("min_conf", 0.25))
    alpha = float(g.get("smoothing_alpha", 0.35))
    global_score_on = float(g.get("score_on", 0.62))
    global_score_off = float(g.get("score_off", 0.42))
    global_hold_on = int(g.get("hold_frames_on", 4))
    global_hold_off = int(g.get("hold_frames_off", 4))
    global_min_len = int(g.get("min_action_frames", 8))
    global_pre_b = int(g.get("pre_buffer_frames", 10))
    global_post_b = int(g.get("post_buffer_frames", 10))
    sequential = bool(g.get("sequential", True))
    skip_idle = bool(g.get("skip_idle", False))
    handoff_prev_end_to_next_start = bool(g.get("handoff_prev_end_to_next_start", False))
    from_tray_debug = bool(g.get("from_tray_debug", False))
    neighbor_fb_cfg = g.get("neighbor_segment_fallback") or {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 0:
        # Safety fallback: some containers may not expose FPS properly.
        fps = 30.0

    static_zones: Dict[str, np.ndarray] = {}
    for name, z in cfg.get("zones", {}).items():
        poly = z.get("polygon_norm", [])
        if poly:
            static_zones[name] = to_abs_poly(poly, w, h)

    by_frame = load_detections_jsonl(det_path, min_conf=min_conf)
    frames = sorted(by_frame.keys())
    if not frames:
        raise SystemExit("No detections found in detections file.")

    actions = cfg.get("actions", [])
    timeline_rows = []
    action_ptr = 0
    scan_start_frame = frames[0]
    pending_actions: List[dict] = []
    # Tray A from the last completed/pending from_tray_pick action (for cross-tray gate).
    last_tray_pick_tray_a_box: Optional[Tuple[float, float, float, float]] = None
    last_tray_pick_tray_a_track_id: Optional[int] = None

    def close_pending_actions(next_start_frame: int, upto_a_idx: Optional[int] = None, close_all: bool = False) -> None:
        """Close pending actions at boundary next_start_frame - 1.

        - close_all=True: flush everything (used at end of video)
        - upto_a_idx: close pending actions whose index is smaller than this action index
        """
        nonlocal pending_actions, timeline_rows
        if not pending_actions:
            return
        end_frame = int(next_start_frame) - 1
        while pending_actions:
            if not close_all:
                pa0 = pending_actions[0]
                if upto_a_idx is None:
                    break
                if int(pa0["a_idx"]) >= int(upto_a_idx):
                    break
            pa = pending_actions.pop(0)
            s = int(pa["start_frame"])
            if end_frame < s:
                continue
            clip_start = max(0, s - int(pa["pre_b"]))
            clip_end = end_frame + int(pa["post_b"])
            clip_name = f"{int(pa['a_idx'])+1:02d}_{pa['act_id']}.mp4"
            cut_clip(video_path, clips_dir / clip_name, clip_start, clip_end)
            timeline_rows.append(
                {
                    "order": int(pa["a_idx"]) + 1,
                    "action_id": pa["act_id"],
                    "description": pa["description"],
                    "start_frame": s,
                    "end_frame": end_frame,
                    "clip_start_frame": clip_start,
                    "clip_end_frame": clip_end,
                    "best_score": round(float(pa["best_score"]), 4),
                    "clip_path": str(clips_dir / clip_name),
                }
            )

    for a_idx, action in enumerate(actions):
        if sequential and a_idx < action_ptr:
            continue
        act_id = str(action.get("id", f"action_{a_idx+1}"))

        # Per-action overrides
        score_on = float(action.get("score_on", global_score_on))
        score_off = float(action.get("score_off", global_score_off))
        hold_on = int(action.get("hold_frames_on", global_hold_on))
        hold_off = int(action.get("hold_frames_off", global_hold_off))
        min_len = int(action.get("min_action_frames", global_min_len))
        pre_b = int(action.get("pre_buffer_frames", global_pre_b))
        post_b = int(action.get("post_buffer_frames", global_post_b))
        max_len = int(action.get("max_action_frames", 0))

        ema = 0.0
        in_action = False
        on_streak = 0
        off_streak = 0
        start_frame = None
        best_score = 0.0

        candidate_frames = [f for f in frames if f >= scan_start_frame]

        # Special mode for action 1: board appears -> board contained in jig for N frames
        if action.get("mode", "") == "board_to_jig_v1":
            board_cls = str(action.get("object_class", "board"))
            jig_cls = str(action.get("target", {}).get("class_name", "jig"))
            board_conf_min = float(action.get("board_conf_min", 0.6))
            jig_conf_min = float(action.get("jig_conf_min", 0.7))
            contain_thr = float(action.get("contain_end_thr", 0.9))
            end_hold = int(action.get("end_hold_frames", 30))
            grace = int(action.get("grace_miss_frames", 2))
            pre_b = int(action.get("pre_buffer_frames", global_pre_b))
            post_b = int(action.get("post_buffer_frames", global_post_b))

            start_frame = None
            end_frame = None
            hold_count = 0
            miss_count = 0
            best_contain = 0.0

            for f in candidate_frames:
                dets = by_frame.get(f, [])
                boards = [d for d in dets if d.cls == board_cls and d.conf >= board_conf_min]
                jigs = [d for d in dets if d.cls == jig_cls and d.conf >= jig_conf_min]

                if start_frame is None and boards:
                    start_frame = f

                if start_frame is None:
                    continue
                if not boards or not jigs:
                    miss_count += 1
                    if miss_count > grace:
                        hold_count = 0
                    continue

                boards.sort(key=lambda d: d.conf, reverse=True)
                jigs.sort(key=lambda d: d.conf, reverse=True)
                b = boards[0]
                j = jigs[0]
                ba = aabb(b.poly)
                ja = aabb(j.poly)
                contain = contain_ratio_aabb(ba, ja)
                best_contain = max(best_contain, contain)
                bcx = (ba[0] + ba[2]) / 2.0
                bcy = (ba[1] + ba[3]) / 2.0
                center_in_jig = 1.0 if (ja[0] <= bcx <= ja[2] and ja[1] <= bcy <= ja[3]) else 0.0

                if contain >= contain_thr and center_in_jig >= 1.0:
                    hold_count += 1
                    miss_count = 0
                else:
                    miss_count += 1
                    if miss_count > grace:
                        hold_count = max(0, hold_count - 1)

                if hold_count >= end_hold:
                    end_frame = f
                    break

            if (
                handoff_prev_end_to_next_start
                and start_frame is not None
                and end_frame is None
                and sequential
            ):
                pending_actions.append(
                    {
                        "a_idx": a_idx,
                        "act_id": act_id,
                        "description": action.get("description", ""),
                        "start_frame": start_frame,
                        "pre_b": pre_b,
                        "post_b": post_b,
                        "best_score": best_contain,
                    }
                )

            if start_frame is not None and end_frame is not None and end_frame >= start_frame:
                if handoff_prev_end_to_next_start and sequential:
                    close_pending_actions(start_frame, upto_a_idx=a_idx)

                clip_start = max(0, start_frame - pre_b)
                clip_end = end_frame + post_b
                clip_name = f"{a_idx+1:02d}_{act_id}.mp4"
                cut_clip(video_path, clips_dir / clip_name, clip_start, clip_end)
                timeline_rows.append(
                    {
                        "order": a_idx + 1,
                        "action_id": act_id,
                        "description": action.get("description", ""),
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "clip_start_frame": clip_start,
                        "clip_end_frame": clip_end,
                        "best_score": round(best_contain, 4),
                        "clip_path": str((clips_dir / clip_name)),
                    }
                )
                if sequential:
                    action_ptr = a_idx + 1
                    scan_start_frame = end_frame + 1
            continue

        # Generic score-based mode for "..._from_tray" pick-like actions.
        # It implements the shared definition:
        #  - Start candidate: "hand center in tray" is large enough AND lasts long enough
        #  - After leaving tray: confirm the "held item(s)" with streak/EMA
        #  - End: end-item is released from hand AND end-item is inside the target
        if action.get("mode", "") == "from_tray_pick_score_v1":
            has_next_tray_pick = any(
                str(nact.get("mode", "")) == "from_tray_pick_score_v1"
                for nact in actions[a_idx + 1 :]
            )
            hand_cls = str(action.get("hand_class", "hand"))
            tray_cls = str(action.get("source", {}).get("class_name", action.get("tray_class", "tray")))
            target_cls = str(action.get("target", {}).get("class_name", action.get("target_class", "board")))

            tray_conf_min = float(action.get("tray_conf_min", 0.7))
            hand_conf_min = float(action.get("hand_conf_min", 0.6))

            # Candidate start: require "hand center in tray"
            hand_tray_iou_min = float(action.get("hand_tray_iou_min", 0.003))
            # Leave detection should be stricter than start detection.
            # Otherwise hand may still be considered "in-tray" even after moving away.
            leave_tray_iou_max = float(action.get("leave_tray_iou_max", hand_tray_iou_min))
            hand_tray_hold_frames = int(action.get("hand_tray_hold_frames", 4))
            leave_hold_frames = int(action.get("leave_hold_frames", 1))
            leave_miss_grace = int(action.get("leave_miss_grace", action.get("grace_miss_frames", 2)))

            # Confirmation after leaving tray
            confirm_item_classes = action.get("confirm_item_classes", action.get("object_class", []))
            if isinstance(confirm_item_classes, str):
                confirm_item_classes = [confirm_item_classes]
            confirm_item_classes = [str(x) for x in confirm_item_classes]

            confirm_iou_min = float(action.get("confirm_hand_iou_min", 0.02))
            confirm_hold_frames = int(action.get("confirm_hold_frames", int(action.get("mid_hold_frames", 2))))
            confirm_miss_grace = int(action.get("confirm_miss_grace", action.get("grace_miss_frames", 2)))

            negative_item_classes = action.get("negative_item_classes", [])
            if isinstance(negative_item_classes, str):
                negative_item_classes = [negative_item_classes]
            negative_item_classes = [str(x) for x in negative_item_classes]
            negative_max_iou = float(action.get("negative_max_hand_iou", 0.05))

            # End item selection & release logic
            end_item_classes = action.get("end_item_classes", confirm_item_classes)
            if isinstance(end_item_classes, str):
                end_item_classes = [end_item_classes]
            end_item_classes = [str(x) for x in end_item_classes]

            end_item_conf_min = float(action.get("end_item_conf_min", float(action.get("object_conf_min", action.get("pcie_conf_min", 0.45)))))
            end_release_base_thr = float(action.get("end_release_hand_iou_max", confirm_iou_min))
            end_target_cover_thr = float(action.get("end_target_cover_thr", float(action.get("target_cover_thr", 0.25))))
            end_target_check_mode = str(action.get("end_target_check_mode", "contain")).lower().strip()
            end_hold = int(action.get("end_hold_frames", 10))
            end_miss_grace = int(action.get("end_miss_grace", action.get("grace_miss_frames", 3)))
            end_hold_mode = str(action.get("end_hold_mode", "consecutive")).lower().strip()
            end_mode = str(action.get("end_mode", "release_only")).lower().strip()
            end_primary_hold = int(action.get("end_primary_hold_frames", end_hold))
            end_primary_timeout = int(action.get("end_primary_timeout_frames", max(60, end_hold * 6)))
            target_conf_min = float(action.get("target_conf_min", action.get("board_conf_min", 0.6)))
            confirm_item_conf_min = float(action.get("confirm_item_conf_min", end_item_conf_min))
            negative_item_conf_min = float(action.get("negative_item_conf_min", confirm_item_conf_min))

            # Optional dynamic release like existing action 3 behavior
            dynamic_release = bool(action.get("dynamic_release", True))
            release_floor_ratio = float(action.get("release_floor_ratio", 0.2))
            release_ratio = float(action.get("release_ratio", 0.25))
            end_require_hand_in_tray = bool(action.get("end_require_hand_in_tray", False))
            end_hand_in_tray_iou_min = float(action.get("end_hand_in_tray_iou_min", hand_tray_iou_min))
            end_exclude_start_tray = bool(action.get("end_exclude_start_tray", False))
            end_exclude_start_tray_iou_thr = float(action.get("end_exclude_start_tray_iou_thr", 0.3))
            cross_tray_hold_frames = int(action.get("cross_tray_hold_frames", end_primary_hold))
            cross_tray_miss_grace = int(action.get("cross_tray_miss_grace", 12))
            cross_tray_contact_iou_min = float(action.get("cross_tray_contact_iou_min", hand_tray_iou_min))

            pre_b = int(action.get("pre_buffer_frames", global_pre_b))
            post_b = int(action.get("post_buffer_frames", global_post_b))

            score_on_local = float(action.get("score_on", score_on))
            score_off_local = float(action.get("score_off", score_off))
            alpha_local = float(action.get("smoothing_alpha", alpha))
            commit_requires_score = bool(action.get("commit_requires_score", False))

            # gasket_on_shielding_v1 end mode params
            gasket_cls = str(action.get("gasket_class", "gasket"))
            tweezers_cls = str(action.get("tweezers_class", "tweezers"))
            gasket_shield_iou_min = float(action.get("gasket_shield_iou_min", 0.10))
            gasket_tweezers_iou_max = float(action.get("gasket_tweezers_iou_max", 0.02))
            gasket_end_hold = int(action.get("gasket_end_hold_frames", 5))
            gasket_min_frames_before_end = int(action.get("gasket_min_frames_before_end", 0))
            gasket_conf_min = float(action.get("gasket_conf_min", 0.25))
            tweezers_conf_min = float(action.get("tweezers_conf_min", 0.25))

            # skip_tray_pick: skip Phase 0/1, start directly at Phase 2
            skip_tray_pick = bool(action.get("skip_tray_pick", False))
            start_from_prev_action_end = bool(action.get("start_from_prev_action_end", False))
            start_from_prev_action_id = str(action.get("start_from_prev_action_id", "")).strip()
            start_from_prev_end_reasons = set(
                str(x) for x in (action.get("start_from_prev_end_reasons", []) or [])
            )

            # Optional strict chaining:
            # this action can start only at (end of previous linked action + 1),
            # and only when previous action completed with allowed end_reason(s).
            if start_from_prev_action_end:
                prev_row = None
                for r in reversed(timeline_rows):
                    if start_from_prev_action_id and str(r.get("action_id", "")) != start_from_prev_action_id:
                        continue
                    if str(r.get("_runtime_outcome", "")) != "completed":
                        continue
                    prev_row = r
                    break
                prev_ok = False
                if prev_row is not None and prev_row.get("end_frame") is not None:
                    prev_end_reason = str(prev_row.get("_runtime_end_reason", ""))
                    reason_ok = (not start_from_prev_end_reasons) or (prev_end_reason in start_from_prev_end_reasons)
                    prev_ok = reason_ok
                if not prev_ok:
                    # strict chaining requested but prerequisite not satisfied:
                    # do not let this action start independently.
                    if from_tray_debug:
                        dbg_dir = out_dir / "debug"
                        dbg_dir.mkdir(parents=True, exist_ok=True)
                        dbg_payload = {
                            "action_id": act_id,
                            "mode": "from_tray_pick_score_v1",
                            "outcome": "blocked",
                            "primary_failure": "start_link_not_satisfied",
                            "completed_segment": False,
                            "start_frame": None,
                            "end_frame": None,
                            "candidate_start_frame": None,
                            "final_phase": 0,
                            "phase2_break_reason": None,
                            "linked_prev_action_id": start_from_prev_action_id,
                            "allowed_prev_end_reasons": sorted(start_from_prev_end_reasons),
                        }
                        (dbg_dir / f"from_tray_{act_id}.json").write_text(
                            json.dumps(dbg_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                    continue
                scan_start_frame = int(prev_row["end_frame"]) + 1
                skip_tray_pick = True

            ema_start = 0.0
            ema_confirm = 0.0
            ema_end = 0.0

            # internal state
            phase = 0  # 0=start-candidate (hand in tray streak), 1=confirm-after-leave, 2=active-mount/end
            candidate_start_frame = None
            candidate_start_best_score = 0.0

            active_hand_track = None
            trayA_box = None
            trayA_track_id: Optional[int] = None
            active_end_track = None
            release_peak_overlap = 0.0

            hand_tray_streak = 0
            leave_miss = 0
            confirm_streak = 0
            confirm_miss = 0
            released_streak = 0
            end_miss = 0
            release_start_frame = None
            release_bad_count = 0
            tray_return_streak = 0
            last_tray_return_frame = None
            use_release_fallback = bool(action.get("use_release_fallback", not has_next_tray_pick))
            phase2_start_frame = None
            end_frame = None

            best_score = 0.0
            phase2_break_reason: Optional[str] = None
            last_scan_frame: Optional[int] = None

            # Option A: next from-tray action only after hand was in a *different* tray long enough vs last pick's tray A.
            prev_tray_a = last_tray_pick_tray_a_box
            prev_tray_tid = last_tray_pick_tray_a_track_id
            need_cross_tray_gate = prev_tray_a is not None
            cross_tray_good = 0
            cross_tray_miss = 0
            cross_tray_gate_ok = not need_cross_tray_gate

            # skip_tray_pick: jump directly to Phase 2 (no tray pick needed)
            if skip_tray_pick:
                phase = 2
                candidate_start_frame = int(scan_start_frame)
                cross_tray_gate_ok = True
                need_cross_tray_gate = False
                phase2_start_frame = int(scan_start_frame)

            candidate_frames_local = [f for f in candidate_frames if f >= int(scan_start_frame)]
            # Track candidate start per frame, but only commit after confirm.
            for f in candidate_frames_local:
                last_scan_frame = f
                dets = by_frame.get(f, [])

                hands = [d for d in dets if d.cls == hand_cls and d.conf >= hand_conf_min]
                trays = [d for d in dets if d.cls == tray_cls and d.conf >= tray_conf_min]
                targets = [d for d in dets if d.cls == target_cls and d.conf >= target_conf_min]

                # Cross-tray gate: run before normal phases until satisfied.
                if need_cross_tray_gate and not cross_tray_gate_ok:
                    if not hands or not trays:
                        cross_tray_good = 0
                        cross_tray_miss = 0
                        continue
                    in_other_tray = False
                    for gh in hands:
                        gate_box = aabb(gh.poly)
                        best_gate_tray, _ = best_single_tray_for_hand(
                            gh,
                            gate_box,
                            trays,
                            cross_tray_contact_iou_min,
                        )
                        if best_gate_tray is not None:
                            t_box = aabb(best_gate_tray.poly)
                            tid = best_gate_tray.track_id
                            # If both track ids exist, they decide same vs different tray (overlapping bboxes OK).
                            # Use IoU vs previous tray A only when track id is missing.
                            if prev_tray_tid is not None and tid is not None:
                                in_other_tray = int(tid) != int(prev_tray_tid)
                            else:
                                in_other_tray = iou_aabb(t_box, prev_tray_a) < end_exclude_start_tray_iou_thr
                        if in_other_tray:
                            break
                    if in_other_tray:
                        cross_tray_good += 1
                        cross_tray_miss = 0
                    else:
                        cross_tray_miss += 1
                        if cross_tray_miss > cross_tray_miss_grace:
                            cross_tray_good = 0
                            cross_tray_miss = 0
                    if cross_tray_good >= cross_tray_hold_frames:
                        cross_tray_gate_ok = True
                    else:
                        continue

                if phase in (0, 1) and (not hands or not trays):
                    continue

                # Choose the best target anchor (conf-aware when available)
                if targets:
                    targets.sort(key=lambda d: getattr(d, "conf", 0.0), reverse=True)
                    target_box = aabb(targets[0].poly)
                else:
                    continue

                trays.sort(key=lambda d: d.conf, reverse=True)

                # Resolve active hand: if we are already tracking, lock it.
                active_hand = None
                hand_box = None
                if hands:
                    if active_hand_track is not None:
                        cand = [h for h in hands if h.track_id == active_hand_track]
                        if cand:
                            # pick highest conf among same track
                            cand.sort(key=lambda d: d.conf, reverse=True)
                            active_hand = cand[0]
                    if active_hand is None and phase in (0, 1):
                        # During tray-pick phases, choose the hand that best matches
                        # a tray interaction instead of the highest-confidence hand.
                        best_hand = None
                        best_iou = -1.0
                        for h in hands:
                            h_box = aabb(h.poly)
                            _tray, iou_v = best_single_tray_for_hand(
                                h,
                                h_box,
                                trays,
                                hand_tray_iou_min,
                            )
                            if _tray is not None and iou_v > best_iou:
                                best_hand = h
                                best_iou = iou_v
                        if best_hand is not None:
                            active_hand = best_hand
                    if active_hand is None:
                        # Fallback (no locked track, no tray-match hand):
                        # choose hand most related to target context.
                        active_hand = max(
                            hands,
                            key=lambda h: iou_aabb(aabb(h.poly), target_box),
                        )
                    hand_box = aabb(active_hand.poly)
                elif phase in (0, 1):
                    continue

                # Choose tray box:
                # - If we already have a locked trayA_box, always evaluate "hand in tray"
                #   against that same tray to prevent tray switching.
                # - Otherwise, pick the first (top-conf) tray that contains the hand center
                #   AND has enough overlap (>= hand_tray_iou_min).
                best_tray_box = None
                best_hand_tray_iou = 0.0
                pick_tray_track_id: Optional[int] = None
                if trayA_box is not None and hand_box is not None:
                    best_tray_box = trayA_box
                    best_hand_tray_iou = iou_aabb(hand_box, best_tray_box)
                elif active_hand is not None and hand_box is not None:
                    best_tray, best_iou = best_single_tray_for_hand(
                        active_hand,
                        hand_box,
                        trays,
                        hand_tray_iou_min,
                    )
                    if best_tray is not None:
                        best_tray_box = aabb(best_tray.poly)
                        best_hand_tray_iou = best_iou
                        pick_tray_track_id = int(best_tray.track_id) if best_tray.track_id is not None else None

                if best_tray_box is None:
                    # Not in tray
                    if phase == 0 and hand_tray_streak > 0:
                        hand_tray_streak = max(0, hand_tray_streak - 1)
                    elif phase == 1:
                        leave_miss += 1
                    continue

                hand_in_tray_ok = best_hand_tray_iou >= hand_tray_iou_min and center_in_aabb(active_hand.poly, best_tray_box)

                # Negative constraint: if negative items overlap the hand too much, don't progress confirm.
                negative_ok = True
                if negative_item_classes and phase in (1, 2):
                    for neg_cls in negative_item_classes:
                        neg_items = [d for d in dets if d.cls == neg_cls and d.conf >= negative_item_conf_min]
                        for it in neg_items:
                            it_box = aabb(it.poly)
                            # Use ANY detected hand overlap to avoid missing negative gating
                            # when the tracked "active_hand" switches.
                            max_neg_ov = 0.0
                            for h in hands:
                                max_neg_ov = max(max_neg_ov, iou_aabb(it_box, aabb(h.poly)))
                            if max_neg_ov > negative_max_iou:
                                negative_ok = False
                                break
                        if not negative_ok:
                            break

                if phase == 0:
                    # Start-candidate based on hand-in-tray streak.
                    if hand_in_tray_ok:
                        hand_tray_streak += 1
                        ema_start = alpha_local * min(1.0, best_hand_tray_iou / max(1e-6, hand_tray_iou_min)) + (1.0 - alpha_local) * ema_start
                        if candidate_start_frame is None:
                            candidate_start_frame = f
                            active_hand_track = active_hand.track_id
                            trayA_box = best_tray_box
                            if pick_tray_track_id is not None:
                                trayA_track_id = pick_tray_track_id
                            candidate_start_best_score = ema_start
                    else:
                        if hand_tray_streak > 0:
                            leave_miss += 1
                            if leave_miss > leave_miss_grace:
                                hand_tray_streak = 0
                                leave_miss = 0
                                active_hand_track = None
                                trayA_box = None
                                trayA_track_id = None
                    if hand_tray_streak >= hand_tray_hold_frames and candidate_start_frame is not None:
                        # Wait for leave stage
                        phase = 1
                        leave_miss = 0
                    continue

                # Phase 1: wait for leave tray and then confirm held item(s)
                if phase == 1:
                    # Detect leaving: hand should be no longer "in tray"
                    # Use iou threshold for leaving instead of the start threshold.
                    if best_hand_tray_iou > leave_tray_iou_max:
                        leave_miss = 0
                        continue
                    else:
                        leave_miss += 1
                        if leave_miss < leave_hold_frames:
                            continue

                    # Confirm after leaving tray
                    if not negative_ok:
                        continue

                    # Compute confirm overlap as max IoU between any confirm item and active_hand.
                    confirm_overlap = 0.0
                    confirm_end_track = None
                    for ccls in confirm_item_classes:
                        citems = [d for d in dets if d.cls == ccls and d.conf >= confirm_item_conf_min]
                        for it in citems:
                            it_box = aabb(it.poly)
                            # Use ANY detected hand overlap to stabilize confirm track selection
                            # when hand tracking switches briefly.
                            ov = 0.0
                            for h in hands:
                                ov = max(ov, iou_aabb(it_box, aabb(h.poly)))
                            if ov > confirm_overlap:
                                confirm_overlap = ov
                                confirm_end_track = it.track_id

                    # update confirm EMA
                    confirm_sig = min(1.0, confirm_overlap / max(1e-6, confirm_iou_min))
                    ema_confirm = alpha_local * confirm_sig + (1.0 - alpha_local) * ema_confirm

                    if confirm_overlap >= confirm_iou_min:
                        confirm_streak += 1
                        confirm_miss = 0
                        best_score = max(best_score, confirm_overlap)
                        release_peak_overlap = max(release_peak_overlap, confirm_overlap)
                    else:
                        confirm_miss += 1
                        if confirm_miss > confirm_miss_grace:
                            # If confirmation is lost too long, restart candidate.
                            phase = 0
                            hand_tray_streak = 0
                            confirm_streak = 0
                            confirm_miss = 0
                            # Keep candidate_start_frame so handoff/pending can still close it
                            # if we already had a valid pick attempt for this action.
                            active_hand_track = None
                            trayA_box = None
                            trayA_track_id = None
                            leave_miss = 0
                            continue
                        continue

                    if confirm_streak >= confirm_hold_frames and (ema_confirm >= score_on_local if commit_requires_score else True):
                        # Commit: action is now active and uses end_item_classes for release.
                        phase = 2
                        phase2_start_frame = f
                        if confirm_end_track is not None:
                            active_end_track = confirm_end_track
                        active_hand_track = active_hand.track_id
                        released_streak = 0
                        end_miss = 0
                    continue

                # Phase 2: active mounting/end
                if phase == 2:
                    # --- gasket_on_shielding_v1: end when gasket is on shielding and off tweezers ---
                    if end_mode == "gasket_on_shielding_v1":
                        gasket_dets = [d for d in dets if d.cls == gasket_cls and d.conf >= gasket_conf_min]
                        shield_dets = [d for d in dets if d.cls == end_item_classes[0] and d.conf >= end_item_conf_min] if end_item_classes else []
                        tweez_dets = [d for d in dets if d.cls == tweezers_cls and d.conf >= tweezers_conf_min]

                        if not gasket_dets or not shield_dets:
                            released_streak = 0
                            continue

                        # Best gasket-shielding IoU
                        best_gs = 0.0
                        for g in gasket_dets:
                            gb = aabb(g.poly)
                            for s in shield_dets:
                                sb = aabb(s.poly)
                                best_gs = max(best_gs, iou_aabb(gb, sb))

                        # Best gasket-tweezers IoU
                        best_gt = 0.0
                        for g in gasket_dets:
                            gb = aabb(g.poly)
                            for t in tweez_dets:
                                tb = aabb(t.poly)
                                best_gt = max(best_gt, iou_aabb(gb, tb))

                        gasket_ok = best_gs >= gasket_shield_iou_min and best_gt < gasket_tweezers_iou_max
                        if gasket_ok:
                            released_streak += 1
                            best_score = max(best_score, best_gs)
                        else:
                            released_streak = 0

                        if released_streak >= gasket_end_hold:
                            # Prevent very early end: require the action to have
                            # progressed enough since phase2 start.
                            min_ok = True
                            if gasket_min_frames_before_end > 0 and phase2_start_frame is not None:
                                min_ok = (f - int(phase2_start_frame) + 1) >= gasket_min_frames_before_end
                            if min_ok:
                                end_frame = f
                                phase2_break_reason = "gasket_on_shielding_met"
                                break

                        # Hard cap
                        if max_len > 0 and candidate_start_frame is not None and (f - candidate_start_frame + 1) >= max_len:
                            end_frame = f
                            phase2_break_reason = "max_action_frames_cap"
                            break
                        continue

                    if end_mode == "tray_return_then_release_fallback" and not use_release_fallback:
                        if active_hand is None or hand_box is None:
                            continue
                        hand_in_any_tray = False
                        end_tray, _ = best_single_tray_for_hand(
                            active_hand,
                            hand_box,
                            trays,
                            end_hand_in_tray_iou_min,
                        )
                        if end_tray is not None:
                            t_box = aabb(end_tray.poly)
                            if end_exclude_start_tray and trayA_box is not None:
                                hand_in_any_tray = iou_aabb(t_box, trayA_box) < end_exclude_start_tray_iou_thr
                            else:
                                hand_in_any_tray = True
                        if hand_in_any_tray:
                            tray_return_streak += 1
                            last_tray_return_frame = f
                        else:
                            tray_return_streak = 0
                        if tray_return_streak >= end_primary_hold:
                            end_frame = f
                            phase2_break_reason = "tray_return_non_fallback_mode"
                            break
                        continue

                    # Select end item instance
                    end_item = None
                    end_overlap = 0.0
                    end_box = None
                    end_candidates = []
                    for ecls in end_item_classes:
                        end_candidates.extend([d for d in dets if d.cls == ecls])

                    if active_end_track is not None:
                        cand = [it for it in end_candidates if it.track_id == active_end_track]
                        if cand:
                            cand.sort(key=lambda d: d.conf, reverse=True)
                            end_item = cand[0]
                    if end_item is None and end_candidates:
                        # fallback by best overlap with active hand
                        for it in end_candidates:
                            it_box = aabb(it.poly)
                            ov = 0.0
                            for h in hands:
                                ov = max(ov, iou_aabb(it_box, aabb(h.poly)))
                            if ov > end_overlap:
                                end_overlap = ov
                                end_item = it
                        if end_item is not None:
                            end_box = aabb(end_item.poly)

                    if end_item is None:
                        end_miss += 1
                        if end_miss > end_miss_grace:
                            # give up; let outer handoff/pending close if next action starts
                            phase2_break_reason = "end_item_lost_after_grace"
                            break
                        continue

                    end_box = aabb(end_item.poly)

                    # Target condition:
                    # - contain: require area coverage inside target zone (contain_ratio)
                    # - hybrid: match special-mode: IoU threshold OR (center in target AND IoU >= 0.1*thr)
                    if end_target_check_mode == "hybrid":
                        end_iou = iou_aabb(end_box, target_box)
                        end_center_ok = center_in_aabb(end_item.poly, target_box)
                        target_ok = (end_iou >= end_target_cover_thr) or (end_center_ok and end_iou >= (0.1 * end_target_cover_thr))
                    else:
                        target_ok = contain_ratio_aabb(end_box, target_box) >= end_target_cover_thr
                    # release overlap signal
                    # Use ANY detected hand overlap to prevent early release
                    # when active_hand tracking switches.
                    end_overlap = 0.0
                    for h in hands:
                        end_overlap = max(end_overlap, iou_aabb(end_box, aabb(h.poly)))
                    # Track maximum overlap so dynamic release threshold
                    # reflects the strongest contact observed after the action starts.
                    release_peak_overlap = max(release_peak_overlap, end_overlap)
                    peak = max(release_peak_overlap, end_overlap)
                    if dynamic_release:
                        dyn_release_thr = max(end_release_base_thr * release_floor_ratio, peak * release_ratio)
                    else:
                        dyn_release_thr = end_release_base_thr

                    # End when overlap is low and item is in target
                    hand_in_any_tray = True
                    if end_require_hand_in_tray:
                        hand_in_any_tray = False
                        if active_hand is not None and hand_box is not None:
                            end_tray, _ = best_single_tray_for_hand(
                                active_hand,
                                hand_box,
                                trays,
                                end_hand_in_tray_iou_min,
                            )
                            if end_tray is not None:
                                t_box = aabb(end_tray.poly)
                                if end_exclude_start_tray and trayA_box is not None:
                                    # Exclude the same tray that we started from (tray A).
                                    # We want a later "hand returned to tray B" behavior.
                                    hand_in_any_tray = iou_aabb(t_box, trayA_box) < end_exclude_start_tray_iou_thr
                                else:
                                    hand_in_any_tray = True

                    # Fallback release condition must stay independent from tray-return condition.
                    released_now = (end_overlap < dyn_release_thr) and target_ok

                    # Document logic:
                    # 1) Prefer ending when hand returns to a different tray long enough.
                    # 2) If that cannot be found in time, fallback to release+target end.
                    if end_mode == "tray_return_then_release_fallback":
                        if hand_in_any_tray:
                            tray_return_streak += 1
                        else:
                            tray_return_streak = 0
                        if tray_return_streak >= end_primary_hold:
                            end_frame = f
                            phase2_break_reason = "tray_return_streak_met"
                            break
                        if phase2_start_frame is None:
                            phase2_start_frame = f
                        if not use_release_fallback:
                            continue
                        if (f - phase2_start_frame + 1) < end_primary_timeout:
                            continue

                    if end_hold_mode == "consecutive":
                        if released_now:
                            released_streak += 1
                            best_score = max(best_score, end_overlap)
                        else:
                            released_streak = 0

                        if released_streak >= end_hold:
                            end_frame = f
                            phase2_break_reason = "release_consecutive_met"
                            break
                    else:
                        # Timeout mode:
                        # Start the timer at the first released_now frame.
                        # Allow some non-released frames up to end_miss_grace.
                        if release_start_frame is None:
                            if released_now:
                                release_start_frame = f
                                release_bad_count = 0
                                best_score = max(best_score, end_overlap)
                        else:
                            if released_now:
                                best_score = max(best_score, end_overlap)
                            else:
                                release_bad_count += 1

                            elapsed = (f - release_start_frame) + 1
                            if release_bad_count > end_miss_grace:
                                # Too many misses/recontacts; restart timer.
                                release_start_frame = None
                                release_bad_count = 0
                            elif elapsed >= end_hold:
                                end_frame = f
                                phase2_break_reason = "release_timer_met"
                                break

                    # Hard cap action length
                    if max_len > 0 and candidate_start_frame is not None and (f - candidate_start_frame + 1) >= max_len:
                        end_frame = f
                        phase2_break_reason = "max_action_frames_cap"
                        break

            if phase == 2 and end_frame is None and phase2_break_reason is None:
                phase2_break_reason = "scan_exhausted_no_end"

            # close pending + output
            if phase == 0:
                start_frame = None
                end_frame = None
            if candidate_start_frame is not None:
                start_frame = int(candidate_start_frame)

            wrote_pending = False
            if (
                handoff_prev_end_to_next_start
                and start_frame is not None
                and end_frame is None
                and sequential
            ):
                # Close any older pending from-tray pick so the new pick starts at a clean boundary.
                if pending_actions:
                    close_pending_actions(int(start_frame), upto_a_idx=a_idx)
                wrote_pending = True
                pending_actions.append(
                    {
                        "a_idx": a_idx,
                        "act_id": act_id,
                        "description": action.get("description", ""),
                        "start_frame": start_frame,
                        "pre_b": pre_b,
                        "post_b": post_b,
                        "best_score": best_score,
                    }
                )
                # Move forward so next action searches for the next pickup cycle,
                # then use handoff boundary to close this pending action.
                action_ptr = a_idx + 1
                next_scan = int(start_frame) + 1
                if last_tray_return_frame is not None:
                    next_scan = max(next_scan, int(last_tray_return_frame) + 1)
                scan_start_frame = next_scan

            if start_frame is not None and end_frame is not None and end_frame >= start_frame:
                if handoff_prev_end_to_next_start and sequential:
                    close_pending_actions(start_frame, upto_a_idx=a_idx)

                clip_start = max(0, start_frame - pre_b)
                clip_end = end_frame + post_b
                clip_name = f"{a_idx+1:02d}_{act_id}.mp4"
                cut_clip(video_path, clips_dir / clip_name, clip_start, clip_end)
                timeline_rows.append(
                    {
                        "order": a_idx + 1,
                        "action_id": act_id,
                        "description": action.get("description", ""),
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "clip_start_frame": clip_start,
                        "clip_end_frame": clip_end,
                        "best_score": round(best_score, 4),
                        "clip_path": str((clips_dir / clip_name)),
                        "segment_source": "from_tray_pick_score_v1",
                        "_runtime_outcome": "completed",
                        "_runtime_end_reason": str(phase2_break_reason or ""),
                    }
                )
                if sequential:
                    action_ptr = a_idx + 1
                    scan_start_frame = end_frame + 1
            # Remember tray A for cross-tray gate on the next from-tray pick action.
            if trayA_box is not None:
                last_tray_pick_tray_a_box = trayA_box
                if trayA_track_id is not None:
                    last_tray_pick_tray_a_track_id = trayA_track_id
            if from_tray_debug:
                dbg_dir = out_dir / "debug"
                dbg_dir.mkdir(parents=True, exist_ok=True)
                completed_ok = (
                    start_frame is not None
                    and end_frame is not None
                    and int(end_frame) >= int(start_frame)
                )
                if completed_ok:
                    outcome = "completed"
                    primary_failure = None
                elif wrote_pending:
                    outcome = "pending_handoff"
                    primary_failure = "no_end_before_next_action_boundary"
                elif need_cross_tray_gate and not cross_tray_gate_ok:
                    outcome = "blocked"
                    primary_failure = "cross_tray_gate_not_satisfied"
                elif phase == 0:
                    outcome = "no_segment"
                    primary_failure = "start_not_committed"
                elif phase == 1:
                    outcome = "no_segment"
                    primary_failure = "confirm_phase_incomplete"
                elif phase == 2:
                    outcome = "no_segment"
                    primary_failure = phase2_break_reason or "phase2_no_end"
                else:
                    outcome = "no_segment"
                    primary_failure = "unknown"
                dbg_payload = {
                    "action_id": act_id,
                    "mode": "from_tray_pick_score_v1",
                    "outcome": outcome,
                    "primary_failure": primary_failure,
                    "completed_segment": completed_ok,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "candidate_start_frame": candidate_start_frame,
                    "final_phase": phase,
                    "phase2_break_reason": phase2_break_reason,
                    "cross_tray": {
                        "needed": need_cross_tray_gate,
                        "opened": cross_tray_gate_ok,
                        "final_good_streak": cross_tray_good,
                    },
                    "scan": {
                        "first_frame": candidate_frames[0] if candidate_frames else None,
                        "last_frame": last_scan_frame,
                        "frame_count": len(candidate_frames),
                    },
                    "thresholds": {
                        "hand_tray_iou_min": hand_tray_iou_min,
                        "leave_tray_iou_max": leave_tray_iou_max,
                        "hand_tray_hold_frames": hand_tray_hold_frames,
                        "confirm_iou_min": confirm_iou_min,
                        "confirm_hold_frames": confirm_hold_frames,
                        "cross_tray_hold_frames": cross_tray_hold_frames,
                        "cross_tray_miss_grace": cross_tray_miss_grace,
                        "cross_tray_contact_iou_min": cross_tray_contact_iou_min,
                        "end_target_cover_thr": end_target_cover_thr,
                        "max_action_frames": max_len,
                    },
                    "wrote_pending": wrote_pending,
                }
                (dbg_dir / f"from_tray_{act_id}.json").write_text(
                    json.dumps(dbg_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            continue

        # Special mode for action 2: hand enters tray -> leave tray and go to board -> pick PCIe near board -> release PCIe on board and hand returns a tray
        if action.get("mode", "") == "pcie_tray_to_board_v1":
            hand_cls = str(action.get("hand_class", "hand"))
            pcie_cls = str(action.get("object_class", "PCIe cable"))
            tray_cls = str(action.get("source", {}).get("class_name", "tray"))
            board_cls = str(action.get("target", {}).get("class_name", "board"))

            hand_conf_min = float(action.get("hand_conf_min", 0.6))
            pcie_conf_min = float(action.get("pcie_conf_min", 0.45))
            tray_conf_min = float(action.get("tray_conf_min", 0.7))
            board_conf_min = float(action.get("board_conf_min", 0.7))
            contact_thr = float(action.get("contact_thr", 0.02))
            board_cover_thr = float(action.get("board_cover_thr", 0.25))
            end_hold = int(action.get("end_hold_frames", 1))
            grace = int(action.get("grace_miss_frames", 2))
            pre_b = int(action.get("pre_buffer_frames", global_pre_b))
            post_b = int(action.get("post_buffer_frames", global_post_b))

            # state 0: wait hand in tray A
            # state 1: hand left tray A, moving toward board (PCIe có thể chưa detect)
            # state 2: PCIe được cầm bởi hand và nằm trên board
            # state 3: release trên board + hand vào tray B, giữ N frames
            state = 0
            start_frame = None
            end_frame = None
            hold_count = 0
            miss_count = 0
            best_score = 0.0
            picked_pcie_track = None
            active_hand_track = None
            trayA_box = None

            for f in candidate_frames:
                dets = by_frame.get(f, [])

                hands = [d for d in dets if d.cls == hand_cls and d.conf >= hand_conf_min]
                pcies = [d for d in dets if d.cls == pcie_cls and d.conf >= pcie_conf_min]
                trays = [d for d in dets if d.cls == tray_cls and d.conf >= tray_conf_min]
                boards = [d for d in dets if d.cls == board_cls and d.conf >= board_conf_min]

                if not trays or not boards:
                    continue
                trays.sort(key=lambda d: d.conf, reverse=True)
                boards.sort(key=lambda d: d.conf, reverse=True)
                board_box = aabb(boards[0].poly)

                hand_in_tray = None
                tray_box_for_hand = None
                # Robust tray selection:
                # In some frames, the top-conf tray is not the one that the hand center is inside.
                # We pick the first tray (by conf) whose AABB contains the hand center.
                for h in hands:
                    # Sort candidate trays by confidence already done above.
                    for t in trays:
                        t_box = aabb(t.poly)
                        # dùng tâm hand nằm trong tray để coi là "vào tray"
                        if center_in_aabb(h.poly, t_box):
                            hand_in_tray = h
                            tray_box_for_hand = t_box
                            break
                    if hand_in_tray is not None:
                        break

                # state 0: start when hand enters tray A
                if state == 0:
                    if hand_in_tray is not None:
                        state = 1
                        start_frame = f
                        active_hand_track = hand_in_tray.track_id
                        trayA_box = tray_box_for_hand
                    continue

                # choose active hand if possible
                active_hand = None
                if active_hand_track is not None:
                    cand = [h for h in hands if h.track_id == active_hand_track]
                    if cand:
                        active_hand = cand[0]
                if active_hand is None and hands:
                    hands.sort(key=lambda d: d.conf, reverse=True)
                    active_hand = hands[0]

                # state 1: hand đã rời tray A, đang trên đường tới board; có thể PCIe chưa detect
                if state == 1:
                    if active_hand is None:
                        continue
                    hand_box = aabb(active_hand.poly)
                    # Check đã thật sự rời tray A (nếu còn)
                    hand_trayA_overlap = iou_aabb(hand_box, trayA_box) if trayA_box is not None else 0.0
                    # Khi tay tiến vào vùng board (overlap nhẹ với board) thì sẵn sàng tìm PCIe
                    hand_near_board = iou_aabb(hand_box, board_box) > contact_thr

                    # Nếu đã xuất hiện PCIe overlap tay và gần board -> coi như đã pick PCIe trên đường
                    pcie_in_hand = None
                    for p in pcies:
                        ov = iou_aabb(aabb(p.poly), hand_box)
                        if ov > contact_thr:
                            pcie_in_hand = p
                            break
                    if hand_trayA_overlap < contact_thr and hand_near_board and pcie_in_hand is not None:
                        picked_pcie_track = pcie_in_hand.track_id
                        state = 2
                    continue

                # resolve active PCIe (prefer picked track)
                active_pcie = None
                if picked_pcie_track is not None:
                    cand = [p for p in pcies if p.track_id == picked_pcie_track]
                    if cand:
                        cand.sort(key=lambda d: d.conf, reverse=True)
                        active_pcie = cand[0]
                if active_pcie is None and pcies:
                    # fallback: pick pcie closest/overlapping to hand
                    if active_hand is not None:
                        hand_box = aabb(active_hand.poly)
                        best = None
                        best_ov = -1.0
                        for p in pcies:
                            ov = iou_aabb(aabb(p.poly), hand_box)
                            if ov > best_ov:
                                best_ov = ov
                                best = p
                        active_pcie = best
                    else:
                        pcies.sort(key=lambda d: d.conf, reverse=True)
                        active_pcie = pcies[0]

                if active_pcie is None:
                    continue

                p_box = aabb(active_pcie.poly)
                in_board = contain_ratio_aabb(p_box, board_box)
                # Use max overlap against ANY detected hand to avoid premature release
                # when active hand tracking switches briefly.
                hand_contact = 0.0
                for h in hands:
                    hand_contact = max(hand_contact, iou_aabb(p_box, aabb(h.poly)))
                best_score = max(best_score, in_board)

                # state 2: hand+PCIe trên board (PCIe vẫn được cầm bởi tay)
                if state == 2:
                    # Đảm bảo tiếp tục có contact và PCIe vẫn nằm trên board
                    if hand_contact > contact_thr and in_board >= board_cover_thr:
                        # đủ điều kiện để sau này release
                        pass
                    # Khi đã thoả cover một lần, cho phép chuyển sang state 3 để bắt release
                    if in_board >= board_cover_thr:
                        state = 3
                        hold_count = 0
                        miss_count = 0
                    continue

                # state 3: kết thúc khi PCIe rời hand (theo yêu cầu mới)
                if state == 3:
                    # Prevent early end if the picked PCIe track disappears (or fallback selects another).
                    if picked_pcie_track is not None and active_pcie.track_id != picked_pcie_track:
                        miss_count += 1
                        if miss_count > grace:
                            hold_count = 0
                            miss_count = 0
                        continue

                    released = hand_contact < contact_thr
                    if released:
                        hold_count += 1
                    else:
                        # PCIe touched/overlapped the hand again => reset released streak
                        hold_count = 0
                    miss_count = 0

                    if hold_count >= end_hold:
                        end_frame = f
                        break

            if (
                handoff_prev_end_to_next_start
                and start_frame is not None
                and end_frame is None
                and sequential
            ):
                # Action started but never satisfied its own end condition.
                # We'll close it at the start_frame of the next successfully-detected action.
                pending_actions.append(
                    {
                        "a_idx": a_idx,
                        "act_id": act_id,
                        "description": action.get("description", ""),
                        "start_frame": start_frame,
                        "pre_b": pre_b,
                        "post_b": post_b,
                        "best_score": best_score,
                    }
                )

            if start_frame is not None and end_frame is not None and end_frame >= start_frame:
                if handoff_prev_end_to_next_start and sequential:
                    close_pending_actions(start_frame, upto_a_idx=a_idx)

                clip_start = max(0, start_frame - pre_b)
                clip_end = end_frame + post_b
                clip_name = f"{a_idx+1:02d}_{act_id}.mp4"
                cut_clip(video_path, clips_dir / clip_name, clip_start, clip_end)
                timeline_rows.append(
                    {
                        "order": a_idx + 1,
                        "action_id": act_id,
                        "description": action.get("description", ""),
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "clip_start_frame": clip_start,
                        "clip_end_frame": clip_end,
                        "best_score": round(best_score, 4),
                        "clip_path": str((clips_dir / clip_name)),
                    }
                )
                if sequential:
                    action_ptr = a_idx + 1
                    scan_start_frame = end_frame + 1
            continue

        # Special mode for action 3(+4): hand picks shielding from tray and mounts to board
        # - start: center of hand in tray
        # - middle: shielding intersects hand (and not PCIe cable intersecting hand)
        # - end: shielding no longer intersects hand AND shielding lies inside board
        if action.get("mode", "") == "shielding_tray_to_board_hand_v1":
            hand_cls = str(action.get("hand_class", "hand"))
            tray_cls = str(action.get("source", {}).get("class_name", "tray"))
            shield_cls = str(action.get("object_class", "shielding"))
            pcie_cls = str(action.get("pcie_class", "PCIe cable"))
            board_cls = str(action.get("target", {}).get("class_name", "board"))

            hand_conf_min = float(action.get("hand_conf_min", 0.6))
            tray_conf_min = float(action.get("tray_conf_min", 0.7))
            shield_conf_min = float(action.get("shielding_conf_min", 0.45))
            pcie_conf_min = float(action.get("pcie_conf_min", 0.45))
            board_conf_min = float(action.get("board_conf_min", 0.7))

            contact_thr = float(action.get("contact_thr_shield_hand", 0.02))
            pcie_hand_max_thr = float(action.get("pcie_hand_max_thr", 0.01))
            hand_tray_start_iou_min = float(action.get("hand_tray_start_iou_min", 0.003))
            shield_release_ratio = float(action.get("shield_release_ratio", 0.25))
            shield_release_floor_ratio = float(action.get("shield_release_floor_ratio", 0.2))
            mid_hold_frames = int(action.get("mid_hold_frames", 5))
            end_hold = int(action.get("end_hold_frames", 1))
            board_cover_thr = float(action.get("board_cover_thr", 0.25))
            grace = int(action.get("grace_miss_frames", 2))

            pre_b = int(action.get("pre_buffer_frames", global_pre_b))
            post_b = int(action.get("post_buffer_frames", global_post_b))

            start_frame = None
            end_frame = None
            state = 0
            active_hand_track = None
            active_shield_track = None
            mid_count = 0
            mid_miss = 0
            end_count = 0
            end_miss = 0
            release_start_frame = None
            release_bad_count = 0
            release_peak_overlap = 0.0  # highest shield-hand overlap during "picked/held" phase
            best_contain = 0.0

            for f in candidate_frames:
                dets = by_frame.get(f, [])

                hands = [d for d in dets if d.cls == hand_cls and d.conf >= hand_conf_min]
                trays = [d for d in dets if d.cls == tray_cls and d.conf >= tray_conf_min]
                shields = [d for d in dets if d.cls == shield_cls and d.conf >= shield_conf_min]
                pcies = [d for d in dets if d.cls == pcie_cls and d.conf >= pcie_conf_min]
                boards = [d for d in dets if d.cls == board_cls and d.conf >= board_conf_min]

                # state 0: start only needs "hand center in tray"
                if state == 0:
                    if not hands or not trays:
                        continue
                    trays.sort(key=lambda d: d.conf, reverse=True)
                    tray_box = aabb(trays[0].poly)
                    hand_start = None
                    for h in hands:
                        hb = aabb(h.poly)
                        # Start when "hand in tray" is detected either by center-in-box
                        # or by non-trivial overlap (more robust to label noise).
                        if center_in_aabb(h.poly, tray_box) or iou_aabb(hb, tray_box) >= hand_tray_start_iou_min:
                            hand_start = h
                            break
                    if hand_start is not None:
                        state = 1
                        active_hand_track = hand_start.track_id
                    continue

                # From state 1 onward, we need shields + boards for meaningful progress
                if not hands or not shields or not boards:
                    continue

                # Use highest-conf board as scene anchor
                boards.sort(key=lambda d: d.conf, reverse=True)
                board_box = aabb(boards[0].poly)

                # Resolve active hand
                active_hand = None
                if active_hand_track is not None:
                    cand = [h for h in hands if h.track_id == active_hand_track]
                    if cand:
                        active_hand = cand[0]

                # Fallback to the highest-confidence hand if the tracked hand disappears.
                # (We still keep shielding track consistency to reduce early-release errors.)
                if active_hand is None and hands:
                    hands.sort(key=lambda d: d.conf, reverse=True)
                    active_hand = hands[0]
                    active_hand_track = active_hand.track_id

                if active_hand is None:
                    continue

                # Select best shielding overlapped with the resolved active_hand.
                # This helps action 3 keep a consistent shielding instance to end on.
                shield_best = None
                shield_hand_iou = 0.0
                if active_hand is not None:
                    active_hand_box = aabb(active_hand.poly)
                    for s in shields:
                        ov = iou_aabb(aabb(s.poly), active_hand_box)
                        if ov > shield_hand_iou:
                            shield_hand_iou = ov
                            shield_best = s

                # Compute how much PCIe overlaps with the active hand.
                # This blocks action 3 from progressing if PCIe is still held by the
                # hand that we are currently tracking for this action.
                pcie_hand_max = 0.0
                if active_hand is not None:
                    active_hand_box = aabb(active_hand.poly)
                    for p in pcies:
                        pcie_hand_max = max(pcie_hand_max, iou_aabb(aabb(p.poly), active_hand_box))

                # Middle phase: after leaving tray, we allow the action
                # to "start" when the hand holds either the target shielding
                # or its attached gasket (tweezers scenario), as long as PCIe
                # is NOT being held.
                #
                # However, for the end-condition we still need actual shielding
                # to be the tracked object, so mid_count/active_shield_track
                # only advance on shielding contact (not gasket-only frames).
                if state == 1:
                    gasket_ready = False
                    gasket_best = None
                    gasket_hand_iou = 0.0

                    # Optional gasket evidence (only for this merged action).
                    # If gasket isn't available, gasket_ready stays False.
                    if shield_cls == "shielding" and "gasket" in action.get("description", "").lower():
                        gasket_cls = str(action.get("gasket_class", "gasket"))
                        gasket_conf_min = float(action.get("gasket_conf_min", shield_conf_min))
                        g_candidates = [d for d in dets if d.cls == gasket_cls and d.conf >= gasket_conf_min]
                        for g in g_candidates:
                            ov = iou_aabb(aabb(g.poly), aabb(active_hand.poly))
                            if ov > gasket_hand_iou:
                                gasket_hand_iou = ov
                                gasket_best = g
                        gasket_ready = gasket_best is not None and gasket_hand_iou >= contact_thr

                    shield_ready = shield_best is not None and shield_hand_iou >= contact_thr
                    pcie_ok = pcie_hand_max <= pcie_hand_max_thr

                    if shield_ready or gasket_ready:
                        # Commit start when PCIe is not dominating.
                        if start_frame is None and pcie_ok:
                            start_frame = f

                        if pcie_ok and shield_ready:
                            # Only advance towards "mounting end" using shielding.
                            if active_shield_track is None:
                                active_shield_track = shield_best.track_id
                            mid_count += 1
                            mid_miss = 0
                        elif not pcie_ok:
                            # If PCIe is held, don't progress towards end.
                            mid_miss = min(grace + 1, mid_miss + 1)
                    else:
                        mid_miss += 1
                        if mid_miss > grace:
                            # reset progress if middle condition breaks too long
                            mid_count = max(0, mid_count - 1)
                            mid_miss = 0

                    if mid_count >= mid_hold_frames:
                        state = 2
                    continue

                # End phase: shielding no longer intersects hand + shielding inside board
                if state == 2:
                    # Resolve active shielding instance
                    active_shield = None
                    if active_shield_track is not None:
                        cand = [s for s in shields if s.track_id == active_shield_track]
                        if cand:
                            # choose highest conf among candidates
                            cand.sort(key=lambda d: d.conf, reverse=True)
                            active_shield = cand[0]
                    if active_shield is None:
                        # If we lost the picked shielding track, do NOT fallback to another
                        # shielding instance; otherwise the "release" condition may be satisfied
                        # by the wrong object => early end.
                        end_miss += 1
                        if end_miss > grace:
                            end_count = 0
                            release_start_frame = None
                            release_bad_count = 0
                            end_miss = 0
                        continue

                    s_box = aabb(active_shield.poly)
                    # "Shielding in board" is often partial in AABB terms,
                    # so we accept either center-in-board or sufficient IoU.
                    # Also re-select the active hand by best overlap with this specific shielding
                    # to avoid accidental hand-track switches causing early "release" end.
                    # For end condition we should be robust to hand track switching:
                    # compute overlap against the hand that overlaps shielding the most
                    # (i.e., "shielding no longer touches the (interacting) hand").
                    best_overlap_hand = None
                    best_overlap = 0.0
                    if hands:
                        for h in hands:
                            ov = iou_aabb(s_box, aabb(h.poly))
                            if ov > best_overlap:
                                best_overlap = ov
                                best_overlap_hand = h

                    board_ok = center_in_aabb(active_shield.poly, board_box) or iou_aabb(s_box, board_box) >= board_cover_thr
                    overlap = best_overlap  # max shielding-hand overlap over all hands

                    # End condition (timeout from first observed "release"):
                    # When shielding is considered released (overlap low + on board),
                    # start a timeout timer; allow short breaks up to `grace`.
                    # Use dynamic threshold based on the peak overlap during holding.
                    # If peak is too small (rare), fall back to a fraction of contact_thr.
                    dyn_release_thr = max(contact_thr * shield_release_floor_ratio, release_peak_overlap * shield_release_ratio)
                    released = (overlap < dyn_release_thr) and board_ok
                    if released:
                        if release_start_frame is None:
                            release_start_frame = f
                        release_bad_count = 0
                    else:
                        if release_start_frame is not None:
                            release_bad_count += 1
                            if release_bad_count > grace:
                                release_start_frame = None
                                release_bad_count = 0

                    best_contain = max(best_contain, iou_aabb(s_box, board_box))

                    if release_start_frame is not None and (f - release_start_frame + 1) >= end_hold:
                        end_frame = f
                        break

            if (
                handoff_prev_end_to_next_start
                and start_frame is not None
                and end_frame is None
                and sequential
            ):
                pending_actions.append(
                    {
                        "a_idx": a_idx,
                        "act_id": act_id,
                        "description": action.get("description", ""),
                        "start_frame": start_frame,
                        "pre_b": pre_b,
                        "post_b": post_b,
                        "best_score": best_contain,
                    }
                )

            if start_frame is not None and end_frame is not None and end_frame >= start_frame:
                if handoff_prev_end_to_next_start and sequential:
                    close_pending_actions(start_frame, upto_a_idx=a_idx)

                clip_start = max(0, start_frame - pre_b)
                clip_end = end_frame + post_b
                clip_name = f"{a_idx+1:02d}_{act_id}.mp4"
                cut_clip(video_path, clips_dir / clip_name, clip_start, clip_end)
                timeline_rows.append(
                    {
                        "order": a_idx + 1,
                        "action_id": act_id,
                        "description": action.get("description", ""),
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "clip_start_frame": clip_start,
                        "clip_end_frame": clip_end,
                        "best_score": round(best_contain, 4),
                        "clip_path": str((clips_dir / clip_name)),
                    }
                )
                if sequential:
                    action_ptr = a_idx + 1
                    scan_start_frame = end_frame + 1
            continue

        # Special mode for action 6 (end of process):
        # start: frame ngay sau action trước (theo sequential scan_start_frame)
        # end: khi không còn thấy class "board" trong N frames liên tiếp (grace)
        if action.get("mode", "") == "board_absent_end_v1":
            board_cls = str(action.get("object_class", "board"))
            board_conf_min = float(action.get("board_conf_min", 0.6))
            jig_cls = str(action.get("source", {}).get("class_name", "jig"))
            jig_conf_min = float(action.get("jig_conf_min", 0.7))
            start_leave_contain_thr = float(action.get("start_leave_contain_thr", 0.9))
            grace = int(action.get("grace_miss_frames", 10))
            pre_b = int(action.get("pre_buffer_frames", global_pre_b))
            post_b = int(action.get("post_buffer_frames", global_post_b))

            start_on_prev_end = bool(action.get("start_on_prev_end", True))
            start_frame = None
            fallback_start_frame = int(scan_start_frame) if start_on_prev_end else None

            had_present = False
            had_board_in_jig = False
            last_present_frame = None
            miss_count = 0
            best_score = 0.0
            end_frame = None

            for f in candidate_frames:
                dets = by_frame.get(f, [])
                boards = [d for d in dets if d.cls == board_cls and d.conf >= board_conf_min]
                jigs = [d for d in dets if d.cls == jig_cls and d.conf >= jig_conf_min]

                if boards:
                    had_present = True
                    miss_count = 0
                    last_present_frame = f
                    best_score = max(best_score, max(d.conf for d in boards))
                    if jigs:
                        boards.sort(key=lambda d: d.conf, reverse=True)
                        jigs.sort(key=lambda d: d.conf, reverse=True)
                        b_box = aabb(boards[0].poly)
                        j_box = aabb(jigs[0].poly)
                        contain = contain_ratio_aabb(b_box, j_box)
                        center_in_jig = center_in_aabb(boards[0].poly, j_box)
                        if contain >= start_leave_contain_thr and center_in_jig:
                            had_board_in_jig = True
                        elif had_board_in_jig and start_frame is None:
                            # Standard start: board has begun leaving jig.
                            start_frame = f
                    continue

                if not had_present:
                    continue

                # board has appeared at least once, now wait for consecutive absence
                miss_count += 1
                if miss_count > grace:
                    end_frame = last_present_frame if last_present_frame is not None else f
                    break

            if start_frame is None and fallback_start_frame is not None and end_frame is not None:
                # Fallback only when standard start condition cannot be detected.
                start_frame = fallback_start_frame

            if start_frame is not None and end_frame is not None and end_frame >= start_frame:
                clip_start = max(0, start_frame - pre_b)
                clip_end = end_frame + post_b
                clip_name = f"{a_idx+1:02d}_{act_id}.mp4"
                cut_clip(video_path, clips_dir / clip_name, clip_start, clip_end)
                timeline_rows.append(
                    {
                        "order": a_idx + 1,
                        "action_id": act_id,
                        "description": action.get("description", ""),
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "clip_start_frame": clip_start,
                        "clip_end_frame": clip_end,
                        "best_score": round(best_score, 4),
                        "clip_path": str((clips_dir / clip_name)),
                    }
                )
                if sequential:
                    action_ptr = a_idx + 1
                    scan_start_frame = end_frame + 1
            continue

        for f in candidate_frames:
            frame_dets = by_frame.get(f, [])
            score, ema, parts = action_score(frame_dets, action, static_zones, ema_prev=ema, alpha=alpha)
            best_score = max(best_score, score)

            if not in_action:
                if score >= score_on:
                    on_streak += 1
                    if on_streak >= hold_on:
                        in_action = True
                        start_frame = f - hold_on + 1
                        off_streak = 0
                else:
                    on_streak = 0
            else:
                # Hard cap action length if configured
                if max_len > 0 and start_frame is not None and (f - start_frame + 1) >= max_len:
                    end_frame = f
                    if end_frame - start_frame + 1 >= min_len:
                        clip_start = max(0, start_frame - pre_b)
                        clip_end = end_frame + post_b
                        clip_name = f"{a_idx+1:02d}_{act_id}.mp4"
                        cut_clip(video_path, clips_dir / clip_name, clip_start, clip_end)
                        timeline_rows.append(
                            {
                                "order": a_idx + 1,
                                "action_id": act_id,
                                "description": action.get("description", ""),
                                "start_frame": start_frame,
                                "end_frame": end_frame,
                                "clip_start_frame": clip_start,
                                "clip_end_frame": clip_end,
                                "best_score": round(best_score, 4),
                                "clip_path": str((clips_dir / clip_name)),
                            }
                        )
                        if sequential:
                            action_ptr = a_idx + 1
                            scan_start_frame = end_frame + 1
                            break
                    in_action = False
                    on_streak = 0
                    off_streak = 0
                    best_score = 0.0
                    continue

                if score <= score_off:
                    off_streak += 1
                    if off_streak >= hold_off:
                        end_frame = f - hold_off
                        if start_frame is not None and end_frame - start_frame + 1 >= min_len:
                            clip_start = max(0, start_frame - pre_b)
                            clip_end = end_frame + post_b
                            clip_name = f"{a_idx+1:02d}_{act_id}.mp4"
                            cut_clip(video_path, clips_dir / clip_name, clip_start, clip_end)
                            timeline_rows.append(
                                {
                                    "order": a_idx + 1,
                                    "action_id": act_id,
                                    "description": action.get("description", ""),
                                    "start_frame": start_frame,
                                    "end_frame": end_frame,
                                    "clip_start_frame": clip_start,
                                    "clip_end_frame": clip_end,
                                    "best_score": round(best_score, 4),
                                    "clip_path": str((clips_dir / clip_name)),
                                }
                            )
                            if sequential:
                                action_ptr = a_idx + 1
                                scan_start_frame = end_frame + 1
                                break
                        in_action = False
                        on_streak = 0
                        off_streak = 0
                        best_score = 0.0
                else:
                    off_streak = 0

    # Flush unresolved pending actions at the earliest next detected action start;
    # only use end-of-video when no later action exists.
    if handoff_prev_end_to_next_start and sequential and pending_actions:
        while pending_actions:
            pa = pending_actions[0]
            pa_order = int(pa["a_idx"]) + 1
            next_starts = [
                int(r["start_frame"])
                for r in timeline_rows
                if int(r.get("order", 0)) > pa_order and int(r.get("start_frame", -1)) >= int(pa["start_frame"])
            ]
            if next_starts:
                close_pending_actions(min(next_starts), upto_a_idx=pa_order + 1)
            else:
                close_pending_actions(int(frames[-1]) + 1, close_all=True)

    # Fill idle segments between detected actions
    timeline_rows.sort(key=lambda x: int(x["start_frame"]))
    fb_rules: List[dict] = []
    if isinstance(neighbor_fb_cfg, list):
        fb_rules = [r for r in neighbor_fb_cfg if isinstance(r, dict)]
    elif isinstance(neighbor_fb_cfg, dict) and neighbor_fb_cfg:
        fb_rules = [neighbor_fb_cfg]
    for fb in fb_rules:
        apply_neighbor_segment_fallback(
            timeline_rows,
            actions,
            fb,
            video_path,
            clips_dir,
            global_pre_b,
            global_post_b,
        )
    timeline_rows.sort(key=lambda x: int(x["start_frame"]))
    if skip_idle:
        # No Idle rows. Also trim clips to [start_frame, end_frame] to avoid overlap caused by buffers.
        final_rows = []
        for r in timeline_rows:
            ns = int(r["start_frame"])
            ne = int(r["end_frame"])
            cp = Path(r["clip_path"]) if r.get("clip_path") else None
            if cp is not None and ne >= ns:
                cut_clip(video_path, cp, ns, ne)
                r["clip_start_frame"] = ns
                r["clip_end_frame"] = ne
            final_rows.append(r)
    else:
        final_rows = []
        if timeline_rows:
            cursor = int(frames[0])
            for r in timeline_rows:
                s = int(r["start_frame"])
                e = int(r["end_frame"])
                if s > cursor:
                    final_rows.append(
                        {
                            "order": 0,
                            "action_id": "Idle",
                            "description": "Idle",
                            "start_frame": cursor,
                            "end_frame": s - 1,
                            "clip_start_frame": cursor,
                            "clip_end_frame": s - 1,
                            "best_score": 0.0,
                            "clip_path": "",
                        }
                    )
                final_rows.append(r)
                cursor = e + 1
            if cursor <= int(frames[-1]):
                final_rows.append(
                    {
                        "order": 0,
                        "action_id": "Idle",
                        "description": "Idle",
                        "start_frame": cursor,
                        "end_frame": int(frames[-1]),
                        "clip_start_frame": cursor,
                        "clip_end_frame": int(frames[-1]),
                        "best_score": 0.0,
                        "clip_path": "",
                    }
                )
        else:
            final_rows = [
                {
                    "order": 0,
                    "action_id": "Idle",
                    "description": "Idle",
                    "start_frame": int(frames[0]),
                    "end_frame": int(frames[-1]),
                    "clip_start_frame": int(frames[0]),
                    "clip_end_frame": int(frames[-1]),
                    "best_score": 0.0,
                    "clip_path": "",
                }
            ]

    # Summary table: total seconds per action_id on the original video.
    # This is computed from [start_frame, end_frame] of each timeline row.
    durations: Dict[str, dict] = {}
    for r in final_rows:
        act_id = str(r.get("action_id", ""))
        if act_id == "Idle":
            continue
        s = int(r.get("start_frame", -1))
        e = int(r.get("end_frame", -1))
        if s < 0 or e < s:
            continue
        seconds = (e - s + 1) / fps
        if act_id not in durations:
            durations[act_id] = {
                "action_id": act_id,
                "description": str(r.get("description", "")),
                "occurrences": 0,
                "total_seconds": 0.0,
            }
        durations[act_id]["occurrences"] += 1
        durations[act_id]["total_seconds"] += float(seconds)

    durations_rows = []
    for act_id, d in durations.items():
        occ = int(d["occurrences"])
        tot = float(d["total_seconds"])
        durations_rows.append(
            {
                "action_id": act_id,
                "description": d.get("description", ""),
                "occurrences": occ,
                "total_seconds": round(tot, 4),
                "mean_seconds": round(tot / occ, 4) if occ > 0 else 0.0,
            }
        )
    durations_rows.sort(key=lambda x: (x["action_id"]))

    action_durations_csv = out_dir / "action_durations.csv"
    with action_durations_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["action_id", "description", "occurrences", "total_seconds", "mean_seconds"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in durations_rows:
            writer.writerow(row)

    if RENDER_TIMER_VIDEO:
        # Render on top of the original video. This does not require clips.
        timer_out = out_dir / "video_with_timer_overlay.mp4"
        render_video_with_timer_overlay(video_path, timer_out, fps, final_rows)

    # Enrich timeline rows with explainability/debug columns.
    action_cfg_by_id = {str(a.get("id", "")): a for a in actions}
    debug_dir = out_dir / "debug"
    for r in final_rows:
        dbg = None
        act_id = str(r.get("action_id", ""))
        seg_source = str(r.get("segment_source", ""))
        # If clips are disabled, clear clip metadata to avoid implying that mp4 files exist.
        if not MAKE_CLIPS:
            if r.get("start_frame") is not None:
                r["clip_start_frame"] = int(r.get("start_frame"))
            if r.get("end_frame") is not None:
                r["clip_end_frame"] = int(r.get("end_frame"))
            r["clip_path"] = ""

        r["quality_score"] = r.get("best_score", "")
        r["outcome"] = "completed"
        r["start_reason"] = "detected_start"
        r["end_reason"] = "detected_end"
        r["debug_file"] = ""

        if act_id == "Idle":
            r["outcome"] = "idle"
            r["start_reason"] = "idle_gap"
            r["end_reason"] = "idle_gap"
            r["reliability_0_100"] = 0.0
            continue

        if seg_source == "neighbor_fallback":
            r["outcome"] = "fallback_segment"
            r["start_reason"] = "neighbor_fallback_prev_end_plus_1"
            r["end_reason"] = "neighbor_fallback_next_start_minus_1"

        dbg_path = debug_dir / f"from_tray_{act_id}.json"
        if dbg_path.exists():
            try:
                dbg = json.loads(dbg_path.read_text(encoding="utf-8"))
            except Exception:
                dbg = {}
            r["debug_file"] = str(dbg_path)
            if dbg:
                r["outcome"] = str(dbg.get("outcome", r["outcome"]))
                primary_failure = dbg.get("primary_failure")
                if primary_failure:
                    r["end_reason"] = str(primary_failure)
                phase2_reason = dbg.get("phase2_break_reason")
                if phase2_reason:
                    r["end_reason"] = str(phase2_reason)
                if dbg.get("candidate_start_frame") is not None:
                    r["start_reason"] = "from_tray_candidate_committed"
                elif dbg.get("start_frame") is not None:
                    r["start_reason"] = "from_tray_start_detected"
                if dbg.get("wrote_pending"):
                    r["end_reason"] = "handoff_prev_end_to_next_start"
                if dbg.get("completed_segment"):
                    r["outcome"] = "completed"

        # Fill default reasons for non-from_tray modes using mode name
        if r.get("start_reason") == "detected_start" or r.get("end_reason") == "detected_end":
            mode = str(action_cfg_by_id.get(act_id, {}).get("mode", "")).strip()
            if mode:
                if r.get("start_reason") == "detected_start":
                    r["start_reason"] = f"{mode}:start"
                if r.get("end_reason") == "detected_end":
                    r["end_reason"] = f"{mode}:end"

        act_mode = str(action_cfg_by_id.get(act_id, {}).get("mode", "")).strip()
        r["reliability_0_100"] = compute_heuristic_reliability_0_100(r, dbg, act_mode)

    # Save timeline
    timeline_csv = out_dir / "timeline.csv"
    with timeline_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "order",
            "action_id",
            "description",
            "start_frame",
            "end_frame",
            "clip_start_frame",
            "clip_end_frame",
            "best_score",
            "quality_score",
            "reliability_0_100",
            "clip_path",
            "segment_source",
            "outcome",
            "start_reason",
            "end_reason",
            "debug_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in final_rows:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)

    timeline_json = out_dir / "timeline.json"
    timeline_json.write_text(json.dumps(final_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    n_actions = sum(1 for r in final_rows if r["action_id"] != "Idle")
    print(f"Done. Found actions: {n_actions}")
    print(f"Timeline CSV: {timeline_csv}")
    print(f"Timeline JSON: {timeline_json}")
    print(f"Clips dir: {clips_dir}")


if __name__ == "__main__":
    main()
