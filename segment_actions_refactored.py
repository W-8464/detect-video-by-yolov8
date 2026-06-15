import argparse
import csv
import json
import numpy as np
import yaml
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

MAKE_CLIPS = False

@dataclass
class Detection:
    cls: str
    conf: float
    poly: np.ndarray
    track_id: Optional[int] = None
    keypoints: Optional[np.ndarray] = None

def aabb(poly: np.ndarray) -> Tuple[float, float, float, float]:
    return float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max())

def aabb_region(poly: np.ndarray, region: Optional[str] = None) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = aabb(poly)
    if region == "bottom_third":
        y1 = y1 + (y2 - y1) * 2.0 / 3.0
    elif region == "top_third":
        y2 = y1 + (y2 - y1) / 3.0
    return x1, y1, x2, y2

def iou_aabb(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0: return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (aa + ab - inter))

def contain_ratio_aabb(inner: Tuple[float, float, float, float], outer: Tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    inner_area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return float(inter / inner_area) if inner_area > 0 else 0.0

def load_detections_jsonl(path: Path, min_conf: float) -> Dict[int, List[Detection]]:
    by_frame: Dict[int, List[Detection]] = {}
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not ln.strip(): continue
        try: row = json.loads(ln)
        except: continue
        frame = int(row.get("frame", -1))
        if frame < 0: continue
        items = row.get("detections", [row])
        by_frame.setdefault(frame, [])
        for d in items:
            cls = str(d.get("class", d.get("cls", "")))
            conf = float(d.get("conf", 0.0))
            poly = d.get("poly", None)
            track_id = d.get("track_id", None)
            if not cls or not poly or conf < min_conf: continue
            kpts_raw = d.get("keypoints", None)
            kpts = np.array(kpts_raw, dtype=np.float32) if kpts_raw else None
            by_frame[frame].append(Detection(cls, conf, np.array(poly), track_id=track_id, keypoints=kpts))
    return by_frame

def cut_clip(video_path: Path, out_path: Path, start_frame: int, end_frame: int) -> None:
    global MAKE_CLIPS
    if not MAKE_CLIPS: return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    cur = start_frame
    while cur <= end_frame:
        ok, frame = cap.read()
        if not ok: break
        writer.write(frame)
        cur += 1
    writer.release()
    cap.release()

def evaluate_condition(dets: List[Detection], cond: dict, class_groups: dict = None, history_frames: List[List[Detection]] = None, locked_track_id: Optional[int] = None) -> bool:
    class_groups = class_groups or {}
    
    c_type = cond.get("type", "")
    if c_type == "or":
        sub_conds = cond.get("sub_conditions", [])
        return any(evaluate_condition(dets, sc, class_groups, history_frames, locked_track_id) for sc in sub_conds)
    if c_type == "and":
        sub_conds = cond.get("sub_conditions", [])
        return all(evaluate_condition(dets, sc, class_groups, history_frames, locked_track_id) for sc in sub_conds)

    def get_classes(name: str) -> List[str]:
        if not name: return []
        return class_groups.get(name, [name])
        
    subj_classes = get_classes(cond.get("subject"))
    tgt_classes = get_classes(cond.get("target"))
    subj_region = cond.get("subject_region")
    tgt_region = cond.get("target_region")

    if c_type == "exists":
        return any(d.cls in subj_classes for d in dets)
        
    if c_type == "missing":
        return not any(d.cls in subj_classes for d in dets)
        
    subjs = sorted([d for d in dets if d.cls in subj_classes], key=lambda x: x.conf, reverse=True)
    if locked_track_id is not None:
        subjs = [d for d in subjs if d.track_id == locked_track_id]
    tgts = sorted([d for d in dets if d.cls in tgt_classes], key=lambda x: x.conf, reverse=True)

    if c_type in ("not_contain", "not_iou"):
        if subjs and not tgts: return True
        if not subjs: return False

    if c_type == "history_missing":
        if not history_frames: return True
        history_len = int(cond.get("history_len", 30))
        past_frames = history_frames[-history_len:] if history_len > 0 else history_frames
        if not past_frames: return True
        missing_count = sum(1 for past_dets in past_frames if not any(d.cls in subj_classes for d in past_dets))
        return missing_count >= int(cond.get("min_missing_frames", history_len))

    if c_type == "history_exists":
        if not history_frames: return False
        history_len = int(cond.get("history_len", 10))
        min_frames = int(cond.get("min_frames", 1))
        past_frames = history_frames[-history_len:] if history_len > 0 else history_frames
        if not past_frames: return False
        exists_count = sum(1 for past_dets in past_frames if any(d.cls in subj_classes for d in past_dets))
        return exists_count >= min_frames

    if c_type == "stationary":
        if not subjs: return False
        if not history_frames: return True
        history_len = int(cond.get("history_len", 10))
        past_frames = history_frames[-history_len:] if history_len > 0 else history_frames
        if not past_frames: return True
        min_iou = float(cond.get("min_iou", 0.85))

        for s in subjs:
            s_box = aabb(s.poly)
            is_stationary = True
            for past_dets in past_frames:
                past_subjs = [pd for pd in past_dets if pd.cls in subj_classes]
                if not past_subjs:
                    is_stationary = False
                    break
                if s.track_id is not None:
                    same_track = [pd for pd in past_subjs if pd.track_id == s.track_id]
                    if same_track:
                        best_iou = max((iou_aabb(s_box, aabb(pd.poly)) for pd in same_track), default=0.0)
                    else:
                        best_iou = max((iou_aabb(s_box, aabb(pd.poly)) for pd in past_subjs), default=0.0)
                else:
                    best_iou = max((iou_aabb(s_box, aabb(pd.poly)) for pd in past_subjs), default=0.0)
                if best_iou < min_iou:
                    is_stationary = False
                    break
            if is_stationary:
                return True
        return False

    if c_type == "not_stationary":
        if not subjs: return True
        if not history_frames: return False
        history_len = int(cond.get("history_len", 10))
        past_frames = history_frames[-history_len:] if history_len > 0 else history_frames
        if not past_frames: return False
        max_iou = float(cond.get("max_iou", 0.70))

        for s in subjs:
            s_box = aabb(s.poly)
            has_moved = True
            for past_dets in past_frames:
                past_subjs = [pd for pd in past_dets if pd.cls in subj_classes]
                if not past_subjs:
                    continue
                if s.track_id is not None:
                    same_track = [pd for pd in past_subjs if pd.track_id == s.track_id]
                    if same_track:
                        best_iou = max((iou_aabb(s_box, aabb(pd.poly)) for pd in same_track), default=0.0)
                    else:
                        best_iou = max((iou_aabb(s_box, aabb(pd.poly)) for pd in past_subjs), default=0.0)
                else:
                    best_iou = max((iou_aabb(s_box, aabb(pd.poly)) for pd in past_subjs), default=0.0)
                if best_iou >= max_iou:
                    has_moved = False
                    break
            if has_moved:
                return True
        return False

    # ── Keypoint-based conditions ──────────────────────────────────────────
    # Keypoint indices (16 total):
    #   0: Wrist
    #   1,2,3: Thumb (CMC, MCP, Tip)
    #   4,5,6: Index (MCP, PIP, Tip)
    #   7,8,9: Middle (MCP, PIP, Tip)
    #   10,11,12: Ring (MCP, PIP, Tip)
    #   13,14,15: Pinky (MCP, PIP, Tip)
    FINGERTIP_IDS = [3, 6, 9, 12, 15]
    WRIST_ID = 0
    MIDDLE_MCP_ID = 7
    THUMB_TIP_ID = 3
    INDEX_TIP_ID = 6

    def _grasp_score(kpts: np.ndarray) -> float:
        """0.0 = fully open, 1.0 = fully closed fist.
        Normalized by wrist-to-middle-mcp distance for scale invariance."""
        if kpts.shape[0] <= max(FINGERTIP_IDS):
            return 0.0
        wrist = kpts[WRIST_ID, :2]
        ref = np.linalg.norm(kpts[MIDDLE_MCP_ID, :2] - wrist)
        if ref < 1.0:
            return 0.0
        dists = []
        for fid in FINGERTIP_IDS:
            conf = kpts[fid, 2] if kpts.shape[1] > 2 else 1.0
            if conf < 0.15:
                continue
            d = np.linalg.norm(kpts[fid, :2] - wrist) / ref
            dists.append(d)
        if not dists:
            return 0.0
        avg = np.mean(dists)
        return float(np.clip(1.0 - (avg - 0.6) / 1.2, 0.0, 1.0))

    if c_type == "grasp":
        if not subjs:
            return False
        min_thr = float(cond.get("min_thr", 0.5))
        for s in subjs:
            if s.keypoints is not None and _grasp_score(s.keypoints) >= min_thr:
                return True
        return False

    if c_type == "release":
        if not subjs:
            return False
        max_thr = float(cond.get("max_thr", 0.35))
        for s in subjs:
            if s.keypoints is not None and _grasp_score(s.keypoints) <= max_thr:
                return True
        return False

    if c_type == "pinch":
        if not subjs:
            return False
        max_thr = float(cond.get("max_thr", 50.0))
        for s in subjs:
            kpts = s.keypoints
            if kpts is None or kpts.shape[0] <= max(THUMB_TIP_ID, INDEX_TIP_ID):
                continue
            c1 = kpts[THUMB_TIP_ID, 2] if kpts.shape[1] > 2 else 1.0
            c2 = kpts[INDEX_TIP_ID, 2] if kpts.shape[1] > 2 else 1.0
            if c1 < 0.15 or c2 < 0.15:
                continue
            d = np.linalg.norm(kpts[THUMB_TIP_ID, :2] - kpts[INDEX_TIP_ID, :2])
            if d <= max_thr:
                return True
        return False

    if not subjs or not tgts: return False
    
    if c_type == "contain":
        for s in subjs:
            s_box = aabb_region(s.poly, subj_region)
            for t in tgts:
                t_box = aabb_region(t.poly, tgt_region)
                if contain_ratio_aabb(s_box, t_box) >= float(cond.get("min_thr", 0.5)):
                    return True
        return False
        
    elif c_type == "not_contain":
        for s in subjs:
            s_box = aabb_region(s.poly, subj_region)
            for t in tgts:
                t_box = aabb_region(t.poly, tgt_region)
                if contain_ratio_aabb(s_box, t_box) >= float(cond.get("max_thr", 0.1)):
                    return False
        return True
        
    elif c_type == "iou":
        for s in subjs:
            s_box = aabb_region(s.poly, subj_region)
            for t in tgts:
                t_box = aabb_region(t.poly, tgt_region)
                if iou_aabb(s_box, t_box) >= float(cond.get("min_thr", 0.05)):
                    return True
        return False
        
    elif c_type == "not_iou":
        for s in subjs:
            s_box = aabb_region(s.poly, subj_region)
            for t in tgts:
                t_box = aabb_region(t.poly, tgt_region)
                if iou_aabb(s_box, t_box) > float(cond.get("max_thr", 0.05)):
                    return False
        return True
        
    return False

def render_video_with_timer_overlay(
    video_path: Path,
    out_path: Path,
    fps: float,
    timeline_rows: List[dict],
) -> None:
    """Render full video (no slicing) with an on-screen timer table overlay."""
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
    rows: List[dict] = []
    max_rows = 10

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

        while seg_idx < len(segments) and frame_idx > segments[seg_idx][1]:
            seg_idx += 1
        cur = segments[seg_idx] if seg_idx < len(segments) else (frame_idx, frame_idx, "idle")
        cur_s, cur_e, cur_id = int(cur[0]), int(cur[1]), str(cur[2])

        last_id = str(rows[-1]["action_id"]) if rows else ""
        if not rows or last_id != cur_id:
            if rows and rows[-1]["action_id"] != "idle":
                prev = rows[-1]
                prev["frozen_seconds"] = (int(prev["end_frame"]) - int(prev["start_frame"]) + 1) / fps_in

            rows.append({
                "action_id": cur_id,
                "start_frame": cur_s,
                "end_frame": cur_e,
                "frozen_seconds": None,
            })
            if len(rows) > max_rows:
                rows = rows[-max_rows:]

        rows[-1]["end_frame"] = cur_e

        lines: List[str] = []
        for r in rows:
            aid = str(r["action_id"])
            if aid == "idle":
                lines.append("idle")
                continue
            frozen = r.get("frozen_seconds")
            elapsed = (frame_idx - int(r["start_frame"]) + 1) / fps_in if frozen is None else float(frozen)
            lines.append(f"{aid}  {elapsed:.2f}s")

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="sample.mp4")
    ap.add_argument("--detections", default="detections_dummy.jsonl")
    ap.add_argument("--config", nargs="+", default=["actions_config_refactored.yaml"],
                    help="Một hoặc nhiều file YAML config. Actions ghép theo thứ tự truyền vào.")
    ap.add_argument("--out-dir", default="runs/actions_refactored")
    ap.add_argument("--make-clips", action="store_true")
    ap.add_argument("--render-timer-video", action="store_true")
    args = ap.parse_args()

    global MAKE_CLIPS
    MAKE_CLIPS = getattr(args, "make_clips", False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    if MAKE_CLIPS: clips_dir.mkdir(parents=True, exist_ok=True)

    config_paths = args.config if isinstance(args.config, list) else [args.config]
    first_cfg = yaml.safe_load(Path(config_paths[0]).read_text(encoding="utf-8"))
    g = first_cfg.get("global", {})
    min_conf = float(g.get("min_conf", 0.25))
    global_pre_b = int(g.get("pre_buffer_frames", 10))
    global_post_b = int(g.get("post_buffer_frames", 10))
    handoff = bool(g.get("handoff_prev_end_to_next_start", False))
    class_groups = dict(g.get("class_groups", {}))

    merged_actions = []
    for p in config_paths:
        _cfg = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
        for k, v in _cfg.get("global", {}).get("class_groups", {}).items():
            if k not in class_groups:
                class_groups[k] = v
        merged_actions.extend(_cfg.get("actions", []))
    if len(config_paths) > 1:
        print(f"[INFO] Merged {len(merged_actions)} actions từ {len(config_paths)} configs: {', '.join(config_paths)}")

    video_path = Path(args.video).resolve()

    if Path(args.detections).exists():
        by_frame = load_detections_jsonl(Path(args.detections), min_conf)
        frames = sorted(by_frame.keys())
    else:
        frames = []

    if not frames:
        print("Không có frame dữ liệu Json.")
        return

    actions = merged_actions
    timeline_rows = []
    
    scan_start_frame = frames[0]
    pending_actions = []

    def close_pending_actions(boundary_frame: int):
        # Resolve any actions stuck in the middle to end up closing right before the next action started.
        while pending_actions:
            pa = pending_actions.pop(0)
            st = int(pa["start_frame"])
            en = boundary_frame - 1
            if en < st: continue
            clip_start = max(0, st - global_pre_b)
            clip_end = en + global_post_b
            clip_name = f"{pa['order']:02d}_{pa['action_id']}.mp4"
            cut_clip(video_path, clips_dir / clip_name, clip_start, clip_end)
            
            pa["end_frame"] = en
            pa["clip_start_frame"] = clip_start
            pa["clip_end_frame"] = clip_end
            pa["clip_path"] = str(clips_dir / clip_name)
            pa["outcome"] = "pending_handoff"
            pa["end_reason"] = "handoff_prev_end_to_next_start"
            timeline_rows.append(pa)

    for a_idx, action in enumerate(actions):
        act_id = str(action.get("id", f"action_{a_idx+1}"))
        phases = action.get("phases", [])
        max_action_frames = int(action.get("max_action_frames", 0))
        force_start = bool(action.get("force_start", False))
        
        current_phase_idx = 0
        streak = 0
        miss_count = 0
        locked_hand_track_id = None
        
        start_frame = None
        end_frame = None
        outcome = "no_segment"
        end_reason = ""

        candidate_frames = [f for f in frames if f >= scan_start_frame]
        
        for f in candidate_frames:
            if force_start and start_frame is None:
                start_frame = f

            # Check hard cap for max_action_frames
            if max_action_frames > 0 and start_frame is not None and (f - start_frame + 1) >= max_action_frames:
                end_frame = f
                outcome = "completed"
                end_reason = "max_action_frames_cap"
                break

            if current_phase_idx >= len(phases):
                end_frame = f - 1
                outcome = "completed"
                end_reason = "naturally_ended"
                break
                
            phase = phases[current_phase_idx]
            conds = phase.get("conditions", [])
            grace = int(phase.get("grace_miss", 2)) # mặc định 2 grace frames
            
            phase_met = True
            if conds:
                h_start = max(0, f - 30)
                hf_list = [by_frame.get(hf, []) for hf in range(h_start, f)]
                phase_met = all(evaluate_condition(by_frame.get(f, []), c, class_groups, history_frames=hf_list, locked_track_id=locked_hand_track_id) for c in conds)
                
            if phase_met:
                streak += 1
                miss_count = 0
                if start_frame is None and current_phase_idx == 0:
                    start_frame = f
                    hand_dets = [d for d in by_frame.get(f, []) if d.cls == "hand"]
                    tray_dets = [d for d in by_frame.get(f, []) if d.cls in class_groups.get("Container", ["tray"])]
                    best_tid = None
                    best_ratio = 0.0
                    for h in hand_dets:
                        h_box = aabb(h.poly)
                        for t in tray_dets:
                            t_box = aabb(t.poly)
                            ratio = contain_ratio_aabb(h_box, t_box)
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_tid = h.track_id
                    locked_hand_track_id = best_tid
                
                hold = int(phase.get("hold_frames", 1))
                if streak >= hold:
                    current_phase_idx += 1
                    streak = 0
                    miss_count = 0
            else:
                if streak > 0 or start_frame is not None:
                    miss_count += 1
                    if miss_count > grace:
                        # Hết grace period, reset streak 
                        streak = 0
                        miss_count = 0
                        # Đối với Phase 0, mất nhãn có nghĩa là chưa hề bắt đầu thực sự
                        if current_phase_idx == 0 and not force_start:
                            start_frame = None
                            
        # Xử lý sau khi quét xong hoặc dính max_frames
        if start_frame is not None:
            if end_frame is None and handoff:
                # Trạng thái "Pending", đẩy vào queue chờ Action tiếp theo đến close.
                pending_actions.append({
                    "order": a_idx + 1,
                    "action_id": act_id,
                    "description": action.get("description", ""),
                    "start_frame": start_frame,
                    "end_frame": None, # Will be set by handoff
                    "outcome": "pending_handoff",
                    "end_reason": ""
                })
                # Set the scan start framework to look ahead so it doesn't get stuck forever
                scan_start_frame = start_frame + 1
            elif end_frame is not None:
                # Hoàn tất bình thường
                if handoff: close_pending_actions(start_frame) # Handoff kill các tác vụ kẹt
                
                clip_start = max(0, start_frame - global_pre_b)
                clip_end = end_frame + global_post_b
                clip_name = f"{a_idx+1:02d}_{act_id}.mp4"
                
                cut_clip(video_path, clips_dir / clip_name, clip_start, clip_end)
                
                timeline_rows.append({
                    "order": a_idx + 1,
                    "action_id": act_id,
                    "description": action.get("description", ""),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "clip_start_frame": clip_start,
                    "clip_end_frame": clip_end,
                    "outcome": outcome,
                    "end_reason": end_reason,
                    "clip_path": str(clips_dir / clip_name)
                })
                
                scan_start_frame = end_frame + 1

    # Cleanup trailing pending actions
    if handoff and pending_actions:
        close_pending_actions(int(frames[-1]) + 1)

    timeline_csv = out_dir / "timeline_refactored_full.csv"
    with open(timeline_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["order", "action_id", "description", "start_frame", "end_frame", "clip_start_frame", "clip_end_frame", "outcome", "end_reason", "clip_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timeline_rows)

    print("================== BÁO CÁO REFACTORED ENGINE ==================")
    print(f"Đã xử lý xong video. Phát hiện {len(timeline_rows)} actions.")
    for row in timeline_rows:
        print(f" - [{row['action_id']:>22}] Frame: {row['start_frame']} -> {row['end_frame']} | Outcome: {row['outcome']} | Lý do: {row['end_reason']}")
    print(f"===============================================================")
    print(f"Kết quả lưu tại: {timeline_csv}")

    if getattr(args, "render_timer_video", False):
        print("\nĐang xuất video render-timer-video...")
        timer_out = out_dir / f"timer_{video_path.name}"
        render_video_with_timer_overlay(video_path, timer_out, 30.0, timeline_rows)
        print(f"Đã lưu timer video tại: {timer_out}")

if __name__ == '__main__':
    main()
