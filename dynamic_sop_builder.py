import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
import cv2

from segment_actions_refactored import Detection, evaluate_condition, aabb, iou_aabb, contain_ratio_aabb


DEFAULT_TUNING = {
    "conf_threshold": 0.55,
    "min_det_conf": 0.25,
    "min_match_frames": 5,
    "max_window_frames": 300,
    "rearm_frames": 12,
    "phase_start_wait_frames": 220,
    "semantic_lookback_frames": 300,
    "attach_anchor_max_gap_frames": 150,
    "gasket_attach_max_gap_frames": 220,
    "min_take_gasket_span_frames": 25,
    "scan_stride": 4,
    "return_role_threshold_offset": 0.1,
}

DEFAULT_CLASS_COLORS = {
    "hand": (0, 255, 0),
    "tray": (255, 0, 0),
    "board": (0, 0, 255),
    "jig": (255, 255, 0),
    "liner": (0, 255, 255),
    "tweezers": (128, 255, 128),
    "shielding": (255, 128, 0),
    "PCIe cable": (255, 0, 255),
    "gasket": (128, 0, 255),
    "bracket": (0, 128, 255),
}

TAKE_GASKET_ACTION_ID = "take_gasket"
APPLY_GASKET_TO_SHIELDING_ACTION_ID = "apply_gasket_to_shielding"
ATTACH_GASKET_PAYLOAD_TO_BOARD_ACTION_ID = "attach_shielding_and_gasket"


def draw_detection_boxes(frame: np.ndarray, detections: List[Detection], class_colors: Optional[Dict[str, tuple]] = None) -> np.ndarray:
    if class_colors is None:
        class_colors = DEFAULT_CLASS_COLORS
    canvas = frame.copy()
    if not detections:
        return canvas

    best_idx: Dict[str, int] = {}
    for i, d in enumerate(detections):
        if d.cls not in best_idx or d.conf > detections[best_idx[d.cls]].conf:
            best_idx[d.cls] = i

    SKELETON = [
        [0, 1], [1, 2], [2, 3],
        [0, 4], [4, 5], [5, 6],
        [0, 7], [7, 8], [8, 9],
        [0, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15]
    ]

    for i, d in enumerate(detections):
        color = class_colors.get(d.cls, (200, 200, 200))
        pts = d.poly.astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

        if d.keypoints is not None and d.cls == "hand":
            kpts = d.keypoints
            for kp in kpts:
                kx, ky = int(kp[0]), int(kp[1])
                conf_kp = kp[2] if kp.shape[0] > 2 else 1.0
                if conf_kp > 0.2:
                    cv2.circle(canvas, (kx, ky), 4, (0, 255, 0), -1)
            for edge in SKELETON:
                p1_idx, p2_idx = edge
                if p1_idx < len(kpts) and p2_idx < len(kpts):
                    p1, p2 = kpts[p1_idx], kpts[p2_idx]
                    c1 = p1[2] if p1.shape[0] > 2 else 1.0
                    c2 = p2[2] if p2.shape[0] > 2 else 1.0
                    if c1 > 0.2 and c2 > 0.2:
                        cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)

        if best_idx.get(d.cls) != i:
            continue

        label = f"{d.cls} {d.conf:.2f}"
        top_idx = int(np.argmin(pts[:, 1]))
        text_x = int(pts[top_idx, 0])
        text_y = max(20, int(pts[top_idx, 1]) - 8)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(
            canvas,
            (text_x - 1, text_y - th - 4),
            (text_x + tw + 4, text_y + baseline + 2),
            color,
            -1,
        )
        cv2.putText(canvas, label, (text_x + 1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def load_detection_frames(detections_jsonl: Path, min_conf: float = 0.0) -> List[List[Detection]]:
    by_frame: Dict[int, List[Detection]] = {}
    max_frame = 0

    for line in detections_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        frame_idx = int(row.get("frame", -1))
        if frame_idx < 0:
            continue
        max_frame = max(max_frame, frame_idx)

        dets = row.get("detections", [])
        parsed: List[Detection] = []
        for d in dets:
            cls_name = str(d.get("class", d.get("cls", ""))).strip()
            if not cls_name:
                continue
            conf = float(d.get("conf", 0.0))
            if conf < min_conf:
                continue
            poly = d.get("poly")
            if not poly:
                continue
            kpts_raw = d.get("keypoints", None)
            kpts = np.array(kpts_raw, dtype=np.float32) if kpts_raw else None
            parsed.append(Detection(cls=cls_name, conf=conf, poly=np.array(poly), track_id=d.get("track_id", None), keypoints=kpts))
        by_frame[frame_idx] = parsed

    if not by_frame:
        return []

    # Normalize to dense frame list [1..max_frame] because detector logs are frame-indexed.
    frames: List[List[Detection]] = []
    for i in range(1, max_frame + 1):
        frames.append(by_frame.get(i, []))
    return frames


class ActionTemplate:
    def __init__(self, yaml_path: Path, tuning: Optional[Dict[str, Any]] = None):
        cfg = load_yaml(yaml_path)
        self.path = yaml_path
        self.global_cfg = cfg.get("global", {})
        self.actions: List[Dict[str, Any]] = cfg.get("actions", [])
        if len(self.actions) != 1:
            raise ValueError(f"Template '{yaml_path}' must contain exactly 1 action")
        self.action = self.actions[0]
        self.base_id = str(self.action.get("id", yaml_path.stem))
        self.repeatable = int(self.action.get("max_action_frames", 1)) == 0
        self._last_detected_component: Optional[str] = None
        _tuning = tuning or {}
        self.phase_start_wait_frames = int(_tuning.get("phase_start_wait_frames", DEFAULT_TUNING["phase_start_wait_frames"]))
        self.semantic_lookback_frames = int(_tuning.get("semantic_lookback_frames", DEFAULT_TUNING["semantic_lookback_frames"]))

    def try_match(
        self,
        frames: List[List[Detection]],
        inherited_class_groups: Dict[str, List[str]],
    ) -> Tuple[int, int, float, int, Optional[set]]:
        phases = self.action.get("phases", [])
        timer_start_phase = int(self.action.get("timer_start_phase", 0))
        if not phases:
            return -1, 0, 0.0, -1, None

        class_groups = dict(inherited_class_groups)
        class_groups.update(self.global_cfg.get("class_groups", {}))

        cursor = 0
        phase_scores: List[float] = []
        actual_start_cursor = -1
        timer_start_cursor = -1

        for p_idx, phase in enumerate(phases):
            hold = max(1, int(phase.get("hold_frames", 1)))
            grace = max(0, int(phase.get("grace_miss", 0)))
            matched_frames = 0
            missed_streak = 0
            phase_start_cursor = cursor

            while cursor < len(frames):
                conds = phase.get("conditions", [])
                history_frames = frames[max(0, cursor - 30):cursor]
                met = all(evaluate_condition(frames[cursor], c, class_groups, history_frames) for c in conds)
                if met:
                    matched_frames += 1
                    missed_streak = 0
                    if matched_frames == 1:
                        if actual_start_cursor == -1:
                            actual_start_cursor = cursor
                        if p_idx == timer_start_phase:
                            timer_start_cursor = cursor
                    if matched_frames >= hold:
                        cursor += 1
                        break
                else:
                    # Allow waiting period before first positive evidence of this phase.
                    if matched_frames == 0:
                        if (cursor - phase_start_cursor) <= self.phase_start_wait_frames:
                            cursor += 1
                            continue
                        return actual_start_cursor, cursor, 0.0, timer_start_cursor, None
                    missed_streak += 1
                    if missed_streak > grace:
                        return actual_start_cursor, cursor, 0.0, timer_start_cursor, None
                cursor += 1

            if matched_frames < hold:
                return actual_start_cursor, cursor, 0.0, timer_start_cursor, None

            # Bo qua chia cho consumed, chi can pass qua host frames se dat confidence tuyet doi
            phase_scores.append(1.0)

        confidence = sum(phase_scores) / max(1, len(phase_scores))
        if timer_start_cursor == -1:
            timer_start_cursor = actual_start_cursor

        hand_track_ids: Optional[set] = None
        if actual_start_cursor >= 0:
            hand_track_ids = set()
            for fi in range(actual_start_cursor, min(len(frames), cursor + 1)):
                for d in frames[fi]:
                    if d.cls == "hand" and d.track_id is not None:
                        hand_track_ids.add(d.track_id)
            if not hand_track_ids:
                hand_track_ids = None

        if not self.verify_component(frames, actual_start_cursor, cursor, inherited_class_groups, hand_track_ids=hand_track_ids):
            return actual_start_cursor, cursor, 0.0, timer_start_cursor, None
        
        return actual_start_cursor, cursor, confidence, timer_start_cursor, hand_track_ids

    def phase_met(
        self,
        frames: List[List[Detection]],
        idx: int,
        inherited_class_groups: Dict[str, List[str]],
        phase_idx: int = 0,
    ) -> bool:
        phases = self.action.get("phases", [])
        if not phases or phase_idx < 0 or phase_idx >= len(phases):
            return False
        phase = phases[phase_idx]
        conds = phase.get("conditions", [])
        class_groups = dict(inherited_class_groups)
        class_groups.update(self.global_cfg.get("class_groups", {}))
        history_frames = frames[max(0, idx - 30):idx]
        return all(evaluate_condition(frames[idx], c, class_groups, history_frames) for c in conds)

    def find_first_phase0_hit(
        self,
        frames: List[List[Detection]],
        inherited_class_groups: Dict[str, List[str]],
        anchor_start_idx: int,
        lookback_frames: Optional[int] = None,
    ) -> Optional[int]:
        if lookback_frames is None:
            lookback_frames = self.semantic_lookback_frames
        left = max(0, anchor_start_idx - lookback_frames)
        right = min(len(frames), anchor_start_idx + 1)
        for idx in range(left, right):
            if self.phase_met(frames, idx, inherited_class_groups, phase_idx=0):
                return idx
        return None

    def find_first_phase0_with_hands(
        self,
        frames: List[List[Detection]],
        inherited_class_groups: Dict[str, List[str]],
        hand_track_ids: set,
        start_idx: int,
        end_idx: int,
    ) -> Optional[int]:
        class_groups = dict(inherited_class_groups)
        class_groups.update(self.global_cfg.get("class_groups", {}))
        phases = self.action.get("phases", [])
        if not phases:
            return None
        phase = phases[0]
        conds = phase.get("conditions", [])
        for idx in range(start_idx, min(end_idx, len(frames))):
            filtered = [
                d for d in frames[idx]
                if d.cls != "hand" or d.track_id in hand_track_ids
            ]
            history_frames = frames[max(0, idx - 30):idx]
            if all(evaluate_condition(filtered, c, class_groups, history_frames) for c in conds):
                return idx
        return None

    def get_detected_component(self) -> Optional[str]:
        return self._last_detected_component

    def verify_component(
        self,
        frames: List[List[Detection]],
        start_frame: int,
        end_frame: int,
        inherited_class_groups: Dict[str, List[str]],
        hand_track_ids: Optional[set] = None,
    ) -> bool:
        """Identify which component was handled by checking:
        1. Component was in Container before/during action start (was available to take)
        2. Component had sustained IoU with hand during the action span
        3. Component left the Container by post-action frames (was actually taken)
        If hand_track_ids is provided, only consider hands with those track_ids.
        Sets self._last_detected_component to the component with highest cumulative IoU."""
        score_rules = self.action.get("score_rules", {})
        verification = score_rules.get("verification", {})

        if not verification or not verification.get("prerequisite", False):
            self._last_detected_component = None
            return True

        component_group = verification.get("component", "Component")
        post_frames = int(verification.get("post_frames", 60))
        min_hits = int(verification.get("min_hits", 1))

        class_groups = dict(inherited_class_groups)
        class_groups.update(self.global_cfg.get("class_groups", {}))

        component_classes = class_groups.get(component_group, [component_group])
        container_classes = class_groups.get("Container", [])

        if start_frame < 0:
            self._last_detected_component = None
            return False

        total_frames = len(frames)
        action_start = max(0, start_frame)
        action_end = min(total_frames, end_frame)
        verify_end = min(total_frames, end_frame + post_frames)

        # Phase A: which components were inside Container at action start?
        # A component that was never in the tray cannot be "taken from tray".
        was_in_container: Dict[str, bool] = {}
        for i in range(action_start, min(total_frames, action_start + 15)):
            frame_dets = frames[i]
            container_dets = [d for d in frame_dets if d.cls in container_classes]
            for c in frame_dets:
                if c.cls not in component_classes:
                    continue
                if c.cls in was_in_container:
                    continue
                if container_dets:
                    c_bbox = aabb(c.poly)
                    if any(contain_ratio_aabb(c_bbox, aabb(ct.poly)) >= 0.30 for ct in container_dets):
                        was_in_container[c.cls] = True
        for comp_cls in component_classes:
            if comp_cls not in was_in_container:
                was_in_container[comp_cls] = False

        # Phase B: accumulate IoU between hand and component during action + post frames
        # Refinement for 'take' actions: identify the specific hand track ID that actually entered the container.
        effective_hand_ids = hand_track_ids
        if hand_track_ids and self.base_id in ("take_component", "take_only", "take_from_liner", "take_liner"):
            hand_scores: Dict[int, float] = {}
            for fi in range(action_start, action_end):
                containers = [d for d in frames[fi] if d.cls in container_classes]
                if not containers: continue
                for d in frames[fi]:
                    if d.cls == "hand" and d.track_id in hand_track_ids:
                        h_box = aabb(d.poly)
                        best_r = max((contain_ratio_aabb(h_box, aabb(ct.poly)) for ct in containers), default=0.0)
                        hand_scores[d.track_id] = hand_scores.get(d.track_id, 0.0) + best_r
            if hand_scores:
                best_tid = max(hand_scores, key=hand_scores.get)
                if hand_scores[best_tid] > 0.1:
                    effective_hand_ids = {best_tid}

        iou_per_class: Dict[str, float] = {}
        hit_frames_per_class: Dict[str, int] = {}
        for i in range(action_start, verify_end):
            frame_dets = frames[i]
            hand_dets = [d for d in frame_dets if d.cls == "hand"]
            if effective_hand_ids is not None:
                hand_dets = [d for d in hand_dets if d.track_id in effective_hand_ids]
            comp_dets = [d for d in frame_dets if d.cls in component_classes]
            container_dets = [d for d in frame_dets if d.cls in container_classes]

            for h in hand_dets:
                h_bbox = aabb(h.poly)
                for c in comp_dets:
                    c_bbox = aabb(c.poly)
                    # In post-action frames, skip component still inside container
                    if i >= end_frame and container_dets:
                        if any(contain_ratio_aabb(c_bbox, aabb(ct.poly)) >= 0.30 for ct in container_dets):
                            continue
                    iou = iou_aabb(h_bbox, c_bbox)
                    if iou > 0.02:
                        iou_per_class[c.cls] = iou_per_class.get(c.cls, 0.0) + iou
                        hit_frames_per_class[c.cls] = hit_frames_per_class.get(c.cls, 0) + 1

        # Phase C: select best component that was in container and has sufficient hits
        eligible = {
            cls: iou for cls, iou in iou_per_class.items()
            if was_in_container.get(cls, False) and hit_frames_per_class.get(cls, 0) >= min_hits
        }

        if eligible:
            self._last_detected_component = max(eligible, key=eligible.get)
            return True

        self._last_detected_component = None
        return False


class DynamicSOPBuilder:
    def __init__(
        self,
        template_paths: List[Path],
        global_cfg: Optional[Dict[str, Any]] = None,
        conf_threshold: Optional[float] = None,
        max_window_frames: Optional[int] = None,
    ):
        self.global_cfg = global_cfg or {}
        _tuning = dict(self.global_cfg.get("tuning", {}))
        _tuning = {k: v for k, v in _tuning.items() if v is not None}

        self.conf_threshold = conf_threshold if conf_threshold is not None else float(_tuning.get("conf_threshold", DEFAULT_TUNING["conf_threshold"]))
        self.min_det_conf = float(_tuning.get("min_det_conf", DEFAULT_TUNING["min_det_conf"]))
        self.min_match_frames = 15 # Increased from 5 to filter noise fragments
        self.max_window_frames = max_window_frames if max_window_frames is not None else int(_tuning.get("max_window_frames", DEFAULT_TUNING["max_window_frames"]))
        self.rearm_frames = int(_tuning.get("rearm_frames", DEFAULT_TUNING["rearm_frames"]))
        self.scan_stride = int(_tuning.get("scan_stride", DEFAULT_TUNING["scan_stride"]))
        self.attach_anchor_max_gap_frames = int(_tuning.get("attach_anchor_max_gap_frames", DEFAULT_TUNING["attach_anchor_max_gap_frames"]))
        self.gasket_attach_max_gap_frames = int(_tuning.get("gasket_attach_max_gap_frames", DEFAULT_TUNING["gasket_attach_max_gap_frames"]))
        self.min_take_gasket_span_frames = int(_tuning.get("min_take_gasket_span_frames", DEFAULT_TUNING["min_take_gasket_span_frames"]))
        return_offset = float(_tuning.get("return_role_threshold_offset", DEFAULT_TUNING["return_role_threshold_offset"]))

        self.role_thresholds = {
            "put": self.conf_threshold,
            "take": self.conf_threshold,
            "attach": self.conf_threshold,
            "return": max(0.4, self.conf_threshold - return_offset),
        }

        _raw_colors = self.global_cfg.get("class_colors", {})
        self.class_colors: Dict[str, tuple] = {
            str(k): tuple(v) for k, v in _raw_colors.items()
        } if _raw_colors else dict(DEFAULT_CLASS_COLORS)

        self.templates = [ActionTemplate(p, tuning=_tuning) for p in template_paths]
        self.template_by_id = {t.base_id: t for t in self.templates}

        self.template_profiles: Dict[str, Dict[str, Any]] = {}
        for t in self.templates:
            phases = t.action.get("phases", [])
            start_conds = phases[0].get("conditions", []) if phases else []
            end_conds = phases[-1].get("conditions", []) if phases else []
            self.template_profiles[t.base_id] = {
                "start_types": {str(c.get("type", "")) for c in start_conds},
                "end_types": {str(c.get("type", "")) for c in end_conds},
                "start_subjects": {str(c.get("subject", "")) for c in start_conds},
                "end_subjects": {str(c.get("subject", "")) for c in end_conds},
            }

    @staticmethod
    def _has_completed_take_gasket_before_attach(
        prior_segments: List[Dict[str, Any]], attach_cand: Dict[str, Any]
    ) -> bool:
        """attach_gasket_to_shielding is only valid after take_gasket has ended earlier in time."""
        attach_start = int(attach_cand["start_frame"])
        for seg in prior_segments:
            tmpl: ActionTemplate = seg["template"]
            if tmpl.base_id != TAKE_GASKET_ACTION_ID:
                continue
            if int(seg["end_frame"]) < attach_start:
                return True
        return False

    @staticmethod
    def _is_generic_take_base_id(base_id: str) -> bool:
        return base_id in ("take_component", "take_only", "take_component_to_target", TAKE_GASKET_ACTION_ID)

    @staticmethod
    def _is_take_anchor_for_generic_attach(base_id: str) -> bool:
        """Any take-like action that should reset generic attach anchoring (excludes naming remap)."""
        return DynamicSOPBuilder._is_generic_take_base_id(base_id) or base_id in (
            "take_liner",
            "take_from_liner",
        )

    @staticmethod
    def _is_generic_attach_base_id(base_id: str) -> bool:
        return base_id in ("attach_component", "attach_only")

    def _gasket_state_before(
        self,
        prior_segments: List[Dict[str, Any]],
        at_frame: int,
    ) -> Dict[str, int]:
        """
        Compute gasket pipeline state since the latest generic attach:
        - latest take_gasket end
        - latest attach_gasket_to_shielding end
        """
        last_generic_attach_end = -1
        for seg in prior_segments:
            tmpl: ActionTemplate = seg["template"]
            seg_start = int(seg["start_frame"])
            seg_end = int(seg["end_frame"])
            if seg_start >= at_frame:
                continue
            if self._is_generic_attach_base_id(tmpl.base_id):
                last_generic_attach_end = max(last_generic_attach_end, seg_end)

        last_take_gasket_end = -1
        last_attach_gasket_end = -1
        for seg in prior_segments:
            tmpl: ActionTemplate = seg["template"]
            seg_start = int(seg["start_frame"])
            seg_end = int(seg["end_frame"])
            if seg_start >= at_frame or seg_end <= last_generic_attach_end:
                continue
            if tmpl.base_id == TAKE_GASKET_ACTION_ID:
                last_take_gasket_end = max(last_take_gasket_end, seg_end)
            elif tmpl.base_id == APPLY_GASKET_TO_SHIELDING_ACTION_ID:
                last_attach_gasket_end = max(last_attach_gasket_end, seg_end)

        return {
            "last_generic_attach_end": last_generic_attach_end,
            "last_take_gasket_end": last_take_gasket_end,
            "last_attach_gasket_end": last_attach_gasket_end,
        }

    def _allow_take_gasket(self, prior_segments: List[Dict[str, Any]], cand: Dict[str, Any]) -> bool:
        st = self._gasket_state_before(prior_segments, int(cand["start_frame"]))
        # Only one take_gasket candidate between two generic attach events.
        return st["last_take_gasket_end"] < 0

    def _allow_attach_gasket(self, prior_segments: List[Dict[str, Any]], cand: Dict[str, Any]) -> bool:
        st = self._gasket_state_before(prior_segments, int(cand["start_frame"]))
        last_take = st["last_take_gasket_end"]
        if last_take < 0:
            return False
        # Only one gasket attach for the pending taken gasket.
        if st["last_attach_gasket_end"] >= last_take:
            return False
        if (int(cand["start_frame"]) - last_take) > self.gasket_attach_max_gap_frames:
            return False
        # Require a shielding to have been taken since the last generic attach/put.
        cand_start = int(cand["start_frame"])
        has_shielding_take = False
        for seg in prior_segments:
            tmpl_seg = seg["template"]
            seg_end = int(seg["end_frame"])
            # Shielding take must be before this attach and after the last cycle reset.
            if seg_end <= st["last_generic_attach_end"] or seg_end >= cand_start:
                continue
            detected = seg.get("detected_component", None)
            if detected == "shielding":
                has_shielding_take = True
                break
        return has_shielding_take

    def _has_take_since_last_generic_attach(
        self,
        prior_segments: List[Dict[str, Any]], attach_cand: Dict[str, Any]
    ) -> bool:
        """
        Generic attach_component should be anchored by at least one completed take
        that happens after the latest completed generic attach in the current path.
        """
        attach_start = int(attach_cand["start_frame"])
        last_attach_end = -1
        for seg in prior_segments:
            tmpl: ActionTemplate = seg["template"]
            seg_end = int(seg["end_frame"])
            if seg_end >= attach_start:
                continue
            if tmpl.base_id in ("attach_component", "attach_only"):
                last_attach_end = max(last_attach_end, seg_end)

        latest_take_end = -1
        for seg in prior_segments:
            tmpl: ActionTemplate = seg["template"]
            seg_end = int(seg["end_frame"])
            if seg_end >= attach_start:
                continue
            if self._is_take_anchor_for_generic_attach(tmpl.base_id):
                if seg_end > last_attach_end:
                    latest_take_end = max(latest_take_end, seg_end)
        if latest_take_end < 0:
            return False
        return (attach_start - latest_take_end) <= self.attach_anchor_max_gap_frames

    @staticmethod
    def _hand_matches_prior_take(
        prior_segments: List[Dict[str, Any]], attach_cand: Dict[str, Any]
    ) -> bool:
        """
        Ensure the attach candidate involves at least one hand that also appeared
        in the most recent take action.  Prevents a hand idling near the target
        from falsely triggering an attach while the real take hand is still busy.
        """
        attach_hand_ids = attach_cand.get("hand_track_ids")
        if not attach_hand_ids:
            return True  # no hand ids to verify → allow (backward compat)

        attach_start = int(attach_cand["start_frame"])
        latest_take = None
        for seg in reversed(prior_segments):
            tmpl = seg["template"]
            if int(seg["end_frame"]) >= attach_start:
                continue
            if DynamicSOPBuilder._is_take_anchor_for_generic_attach(tmpl.base_id):
                latest_take = seg
                break

        if latest_take is None:
            return False

        take_hand_ids = latest_take.get("hand_track_ids")
        if not take_hand_ids:
            return True

        return bool(attach_hand_ids & take_hand_ids)

    def _get_template_for_role(self, role: str) -> ActionTemplate:
        role_map = {
            "put": ("put_board_into_jig", "put_board"),
            "take": ("take_component", "take_only"),
            "attach": ("attach_component", "attach_only"),
            "return": ("board_back_to_conveyor", "return_board"),
        }
        candidates = role_map.get(role, ())
        for c in candidates:
            if c in self.template_by_id:
                return self.template_by_id[c]
        raise ValueError(f"Missing template for role '{role}'. Candidates: {candidates}")

    def _find_next_rising_edge(
        self,
        phase0_flags: List[bool],
        start_idx: int,
        min_false_before: int,
    ) -> Optional[int]:
        false_streak = min_false_before
        if start_idx > 0:
            # Rebuild false streak from nearby history.
            false_streak = 0
            j = start_idx - 1
            while j >= 0 and not phase0_flags[j]:
                false_streak += 1
                j -= 1
        for i in range(start_idx, len(phase0_flags)):
            if phase0_flags[i]:
                prev_true = i > 0 and phase0_flags[i - 1]
                if not prev_true and false_streak >= min_false_before:
                    return i
                false_streak = 0
            else:
                false_streak += 1
        return None

    def _template_role(self, tmpl: ActionTemplate) -> str:
        bid = tmpl.base_id
        if bid in ("put_board_into_jig", "put_board"):
            return "put"
        if bid in (
            "take_component_to_target",
            "take_component",
            "take_only",
            "take_from_liner",
            "take_liner",
            TAKE_GASKET_ACTION_ID,
        ):
            return "take"
        if bid in ("attach_component", "attach_only"):
            return "attach"
        if bid == APPLY_GASKET_TO_SHIELDING_ACTION_ID:
            return "subtask"
        if bid in ("board_back_to_conveyor", "return_board"):
            return "return"
        return "other"

    @staticmethod
    def _is_repeatable_role(role: str) -> bool:
        return role in ("take", "attach")

    def _node_score(self, cand: Dict[str, Any]) -> float:
        duration = max(1, int(cand["end_frame"]) - int(cand["start_frame"]) + 1)
        dur_bonus = min(0.25, duration / 600.0)
        role = self._template_role(cand["template"])
        # Use timer_start_frame for role bias to be more accurate about when action happens
        start = int(cand.get("timer_start_frame", cand["start_frame"]))
        role_bias = 0.0
        # if role == "put":
        #     if start <= 80:
        #         role_bias = 1.2
        #     elif start <= 400:
        #         role_bias = 0.8
        #     elif start <= 800:
        #         role_bias = 0.5
        #     else:
        #         role_bias = 0.2
        # elif role == "return":
        #     if start >= 620:
        #         role_bias = 0.6 
        if role in ("take", "attach", "take_attach", "take_only"):
            role_bias = 0.5
        return float(cand["confidence"]) + dur_bonus + role_bias

    @staticmethod
    def _ordering_start(cand: Dict[str, Any]) -> int:
        """
        Prefer timer_start_frame (actual action) over semantic/window start.
        This allows actions to occur during the 'waiting/stable' phases of other actions
        (like put_board or return_board) without being penalized as overlaps.
        """
        return int(cand.get("timer_start_frame", cand.get("semantic_start_frame", cand["start_frame"])))

    def _transition_score(self, prev: Dict[str, Any], cur: Dict[str, Any]) -> float:
        gap = self._ordering_start(cur) - int(prev["end_frame"])
        if gap < -45:
            return -3.0
        
        score = 0.0
        if gap < 0:
            score -= (-gap / 45.0) * 2.0
        else:
            score -= min(1.5, gap / 140.0)

        prev_t: ActionTemplate = prev["template"]
        cur_t: ActionTemplate = cur["template"]
        prev_role = self._template_role(prev_t)
        cur_role = self._template_role(cur_t)

        # Discourage unreasonable self-loop except repeatable template.
        if prev_t.base_id == cur_t.base_id:
            if cur_t.repeatable or self._is_repeatable_role(cur_role):
                # Small penalty to encourage movement, but allow repetition for components
                score -= 0.1
            else:
                score -= 3.0

        # Soft priors (not hard order constraints).
        if prev_role == "put" and cur_role == "take":
            score += 0.6
        elif prev_role == "take" and cur_role == "take":
            score -= 1.0
            # Supply / special pickups (liner, gasket, from-liner) often follow a component take.
            if prev_t.base_id != cur_t.base_id:
                if cur_t.base_id in ("take_liner", TAKE_GASKET_ACTION_ID, "take_from_liner"):
                    score += 1.05
                elif prev_t.base_id in ("take_liner", "take_from_liner"):
                    score += 0.95
        elif prev_role == "take" and cur_role == "attach":
            score += 0.6
        elif prev_role == "attach" and cur_role == "take":
            score += 0.5
            if cur_t.base_id == "take_liner":
                score += 0.35
        elif prev_role == "attach" and cur_role == "attach":
            score += 0.35
        elif prev_role == "attach" and cur_role == "return":
            score += 0.6
        elif prev_role == "take" and cur_role == "put":
            score -= 1.4
        elif prev_role == "take" and cur_role == "return":
            score += 0.4
        elif prev_role == "return" and cur_role == "put":
            # Strongly encourage return -> put, but keep gap penalty
            # so nearby puts are preferred over distant ones.
            gap = self._ordering_start(cur) - int(prev["end_frame"])
            if gap < -45:
                return -3.0
            score -= min(1.5, max(0, gap) / 140.0)
            score += 3.5
        elif prev_role == "put" and cur_role == "return":
            score -= 0.5

        if prev_t.base_id == TAKE_GASKET_ACTION_ID and cur_t.base_id == APPLY_GASKET_TO_SHIELDING_ACTION_ID:
            score += 0.75

        return score

    def _temporal_iou(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        a0, a1 = int(a["start_frame"]), int(a["end_frame"])
        b0, b1 = int(b["start_frame"]), int(b["end_frame"])
        inter = max(0, min(a1, b1) - max(a0, b0) + 1)
        if inter <= 0:
            return 0.0
        ua = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
        return float(inter / max(1, ua))

    def _collect_candidates(
        self,
        frames: List[List[Detection]],
        inherited_groups: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for tmpl in self.templates:
            role = self._template_role(tmpl)
            thr = self.role_thresholds.get(role, self.conf_threshold)
            local: List[Dict[str, Any]] = []
            for start in range(0, len(frames), self.scan_stride):
                # Local check for tweezers if applicable to this template
                if tmpl.base_id == TAKE_GASKET_ACTION_ID:
                    window = frames[start : start + self.max_window_frames]
                    has_tweezers = any(any(det.cls == "tweezers" for det in f) for f in window)
                    if not has_tweezers:
                        continue

                start_rel, end_rel, conf, timer_rel, hand_ids = tmpl.try_match(
                    frames[start : start + self.max_window_frames], inherited_groups
                )
                if end_rel <= 0 or conf < thr:
                    continue
                end_abs = min(len(frames), start + end_rel)
                semantic_start = start + start_rel + 1 if start_rel >= 0 else start + 1
                match_duration = end_abs - semantic_start + 1
                if match_duration < self.min_match_frames:
                    continue
                timer_start = start + timer_rel + 1 if timer_rel >= 0 else semantic_start
                detected_comp = tmpl.get_detected_component()
                if detected_comp is None and tmpl.base_id == TAKE_GASKET_ACTION_ID:
                    detected_comp = "gasket"
                local.append(
                    {
                        "template": tmpl,
                        "start_frame": start + 1,
                        "semantic_start_frame": semantic_start,
                        "timer_start_frame": timer_start,
                        "end_frame": end_abs,
                        "confidence": round(conf, 4),
                        "detected_component": detected_comp,
                        "hand_track_ids": hand_ids,
                    }
                )

            # Per-template temporal NMS to avoid dense duplicates.
            local.sort(key=lambda x: (x["confidence"], x["end_frame"] - x["start_frame"]), reverse=True)
            kept: List[Dict[str, Any]] = []
            for cand in local:
                if any(self._temporal_iou(cand, k) > 0.7 for k in kept):
                    continue
                kept.append(cand)
                
            # Anchor Logic: Hardcode boundaries for put and return
            # if kept:
            #     if role == "put":
            #         kept = [min(kept, key=lambda x: int(x["start_frame"]))]
            #     elif role == "return":
            #         kept = [max(kept, key=lambda x: int(x["start_frame"]))]

            candidates.extend(sorted(kept, key=lambda x: x["start_frame"]))

        return sorted(candidates, key=lambda x: (x["start_frame"], x["end_frame"]))

    def _decode_best_path(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        n = len(candidates)
        dp = [self._node_score(c) for c in candidates]
        prev_idx = [-1] * n

        for i in range(n):
            for j in range(i):
                edge = self._transition_score(candidates[j], candidates[i])
                if edge <= -2.8:
                    continue
                val = dp[j] + edge + self._node_score(candidates[i])
                if val > dp[i]:
                    dp[i] = val
                    prev_idx[i] = j

        best = max(range(n), key=lambda i: dp[i])
        path: List[Dict[str, Any]] = []
        cur = best
        while cur != -1:
            path.append(candidates[cur])
            cur = prev_idx[cur]
        path.reverse()
        return path

    def _is_reset_candidate(self, cand: Dict[str, Any]) -> bool:
        tmpl: ActionTemplate = cand["template"]
        p = self.template_profiles.get(tmpl.base_id, {})
        end_types = p.get("end_types", set())
        return bool({"missing", "not_contain", "not_iou"} & end_types)

    def _decode_best_path_stateful(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        beam_width = 64
        max_steps = 100  # Increased from 18 to handle more cycles and actions
        # frozen_nonrepeat: non-repeatable action ids seen since last reset
        beams: List[Dict[str, Any]] = [
            {
                "score": 0.0,
                "last_idx": -1,
                "last_end": 0,
                "frozen_nonrepeat": frozenset(),
                "path": [],
            }
        ]

        for _ in range(max_steps):
            expanded: List[Dict[str, Any]] = []
            for st in beams:
                # Keep current state as candidate terminal path.
                expanded.append(st)
                start_i = st["last_idx"] + 1
                for i in range(start_i, len(candidates)):
                    cand = candidates[i]
                    ord_s = self._ordering_start(cand)
                    if ord_s <= int(st["last_end"]) - 45:
                        continue

                    tmpl: ActionTemplate = cand["template"]

                    if tmpl.base_id == APPLY_GASKET_TO_SHIELDING_ACTION_ID:
                        prior = [candidates[j] for j in st["path"]]
                        if ord_s <= int(st["last_end"]):
                            continue
                        if not self._allow_attach_gasket(prior, cand):
                            continue
                    elif tmpl.base_id == TAKE_GASKET_ACTION_ID:
                        prior = [candidates[j] for j in st["path"]]
                        if ord_s <= int(st["last_end"]):
                            continue
                        if (
                            int(cand["end_frame"]) - int(cand["start_frame"]) + 1
                            < self.min_take_gasket_span_frames
                        ):
                            continue
                        if not self._allow_take_gasket(prior, cand):
                            continue
                    elif tmpl.base_id in ("attach_component", "attach_only"):
                        prior = [candidates[j] for j in st["path"]]
                        if ord_s <= int(st["last_end"]):
                            continue
                        gasket_state = self._gasket_state_before(prior, int(cand["start_frame"]))
                        # If a gasket is taken in current cycle, require gasket->shielding attach first.
                        if (
                            gasket_state["last_take_gasket_end"] >= 0
                            and gasket_state["last_attach_gasket_end"] < gasket_state["last_take_gasket_end"]
                        ):
                            continue
                        if not self._has_take_since_last_generic_attach(prior, cand):
                            continue
                        if not self._hand_matches_prior_take(prior, cand):
                            continue
                    elif tmpl.base_id == ATTACH_GASKET_PAYLOAD_TO_BOARD_ACTION_ID:
                        prior = [candidates[j] for j in st["path"]]
                        if ord_s <= int(st["last_end"]):
                            continue
                        gasket_state = self._gasket_state_before(prior, int(cand["start_frame"]))
                        if gasket_state["last_attach_gasket_end"] < 0:
                            continue
                        # Ensure shielding was taken
                        has_shielding = any(
                            (p["template"].base_id in ("take_only", "take_component") and p.get("detected_component") == "shielding")
                            or p["template"].base_id == "take_shielding"
                            for p in prior
                        )
                        if not has_shielding:
                            continue

                    # Force "put" as the starting action if a "put" candidate exists in pool.
                    if not st["path"] and self._template_role(tmpl) != "put":
                        if any(self._template_role(c["template"]) == "put" for c in candidates):
                            continue
                            
                    bid = tmpl.base_id
                    frozen = st["frozen_nonrepeat"]
                    if (not tmpl.repeatable) and (bid in frozen):
                        continue

                    # Block consecutive return_board actions (prevent duplication)
                    if st["path"]:
                        prev = candidates[st["path"][-1]]
                        prev_role = self._template_role(prev["template"])
                        cur_role = self._template_role(tmpl)
                        if prev_role == "return" and cur_role == "return":
                            continue

                    edge = 0.0
                    if st["path"]:
                        prev = candidates[st["path"][-1]]
                        edge = self._transition_score(prev, cand)
                        if edge <= -2.8:
                            continue

                    new_frozen = frozen
                    if self._is_reset_candidate(cand):
                        new_frozen = frozenset()
                    if not tmpl.repeatable:
                        new_frozen = frozenset(set(new_frozen) | {bid})

                    expanded.append(
                        {
                            "score": float(st["score"]) + edge + self._node_score(cand),
                            "last_idx": i,
                            "last_end": int(cand["end_frame"]),
                            "frozen_nonrepeat": new_frozen,
                            "path": st["path"] + [i],
                        }
                    )

            # Deduplicate near-equivalent states and keep top beam.
            dedup: Dict[Tuple[int, int, Tuple[str, ...]], Dict[str, Any]] = {}
            for st in expanded:
                key = (
                    int(st["last_idx"]),
                    int(st["last_end"]),
                    tuple(sorted(st["frozen_nonrepeat"])),
                )
                if key not in dedup or float(st["score"]) > float(dedup[key]["score"]):
                    dedup[key] = st
            beams = sorted(dedup.values(), key=lambda x: float(x["score"]), reverse=True)[:beam_width]

            # If nothing expanded beyond empty paths, stop.
            if all(len(st["path"]) == 0 for st in beams):
                break

        best = max(beams, key=lambda x: (float(x["score"]), len(x["path"])))
        return [candidates[i] for i in best["path"]]

    def _augment_intermediate_candidates(
        self,
        path: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not path:
            return path
        
        # Identify all (put, return) pairs in the path
        intervals: List[Tuple[int, int]] = []
        last_put_end = -1
        
        for p in path:
            role = self._template_role(p["template"])
            if role == "put":
                last_put_end = int(p["end_frame"])
            elif role == "return" and last_put_end != -1:
                intervals.append((last_put_end + 1, self._ordering_start(p) - 1))
                last_put_end = -1
        
        if not intervals:
            # Fallback to first put and last return if no clean pairs
            put_pos = [i for i, p in enumerate(path) if self._template_role(p["template"]) == "put"]
            ret_pos = [i for i, p in enumerate(path) if self._template_role(p["template"]) == "return"]
            if put_pos and ret_pos:
                intervals = [(int(path[put_pos[0]]["end_frame"]) + 1, self._ordering_start(path[ret_pos[-1]]) - 1)]
            elif put_pos:
                intervals = [(int(path[put_pos[0]]["end_frame"]) + 1, 10**9)]
            else:
                return path

        selected = list(path)
        existing_ranges = [(self._ordering_start(p), int(p["end_frame"])) for p in selected]
        
        # Only consider take/attach candidates for augmentation
        pool = [
            c for c in candidates 
            if self._template_role(c["template"]) in ("take", "attach", "take_attach", "take_only", "subtask")
            and float(c["confidence"]) >= max(0.5, self.role_thresholds.get(self._template_role(c["template"]), 0.5) - 0.05)
        ]
        pool.sort(key=lambda x: (x["start_frame"], -x["confidence"]))

        for cand in pool:
            s, e = int(cand["start_frame"]), int(cand["end_frame"])
            
            # Must fall into one of the valid board intervals
            in_interval = any(s >= i_s and e <= i_e for i_s, i_e in intervals)
            if not in_interval:
                continue
                
            # Must not overlap with existing actions in path
            if any(not (e + self.rearm_frames < es or s >= ee + self.rearm_frames) for es, ee in existing_ranges):
                continue

            # State checks for special items (gaskets etc)
            if cand["template"].base_id == APPLY_GASKET_TO_SHIELDING_ACTION_ID:
                if not self._allow_attach_gasket(selected, cand): continue
            elif cand["template"].base_id == TAKE_GASKET_ACTION_ID:
                if not self._allow_take_gasket(selected, cand): continue
            elif cand["template"].base_id in ("attach_component", "attach_only"):
                if not self._has_take_since_last_generic_attach(selected, cand): continue
                if not self._hand_matches_prior_take(selected, cand): continue
            elif cand["template"].base_id == ATTACH_GASKET_PAYLOAD_TO_BOARD_ACTION_ID:
                gasket_state = self._gasket_state_before(selected, int(cand["start_frame"]))
                if gasket_state["last_attach_gasket_end"] < 0:
                    continue
                # Ensure shielding take
                has_shielding = any(
                    (p["template"].base_id in ("take_only", "take_component") and p.get("detected_component") == "shielding")
                    or p["template"].base_id == "take_shielding"
                    for p in selected
                )
                if not has_shielding:
                    continue

            selected.append(cand)
            existing_ranges.append((s, e))

        selected.sort(key=lambda x: x["start_frame"])
        return selected

    def _attach_return_candidate(
        self,
        path: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not path:
            return path
        if any(self._template_role(p["template"]) == "return" for p in path):
            return path

        last_end = max(int(p["end_frame"]) for p in path)
        return_pool = [
            c
            for c in candidates
            if self._template_role(c["template"]) == "return"
            and int(c["start_frame"]) >= (last_end - 40)
            and float(c["confidence"]) >= self.role_thresholds["return"]
        ]
        if not return_pool:
            return path
        # Prefer latest reliable return candidate.
        best = max(return_pool, key=lambda c: (int(c["start_frame"]), float(c["confidence"])))
        out = list(path)
        out.append(best)
        out.sort(key=lambda x: x["start_frame"])
        return out

    def _trim_generic_attach_before_next_take(
        self,
        path: List[Dict[str, Any]],
        frames: Optional[List[List[Detection]]] = None,
        inherited_groups: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Trim generic attach tails if a subsequent take starts, so attach does not
        bleed into the next pickup action across different workstations/templates.

        Option B: When *frames* data is available, scan the attach's frame range
        for the earliest phase-0 hit of the next take template.  If found earlier
        than the current take's start_frame, clip the attach at that boundary and
        re-match the take template from there for tighter boundaries.
        """
        if not path:
            return path
        out = [dict(p) for p in path]
        for i in range(len(out) - 1):
            cur = out[i]
            nxt = out[i + 1]
            cur_t: ActionTemplate = cur["template"]
            nxt_t: ActionTemplate = nxt["template"]
            if cur_t.base_id not in ("attach_component", "attach_only"):
                continue
            if self._template_role(nxt_t) != "take":
                continue
            nxt_start = self._ordering_start(nxt)
            cur_start = int(cur["start_frame"])
            cur_end = int(cur["end_frame"])
            if nxt_start <= cur_start:
                continue

            # --- Option B: scan attach range for early phase0 hit of next take ---
            if frames is not None and inherited_groups is not None:
                # Skip a small offset from attach start to avoid boundary noise.
                min_offset = 15
                scan_from = max(0, cur_start - 1 + min_offset)  # 0-based index
                scan_to = min(len(frames), cur_end)              # exclusive

                early_phase0_frame = None  # will be 1-based if found
                for idx in range(scan_from, scan_to):
                    if nxt_t.phase_met(frames, idx, inherited_groups, phase_idx=0):
                        early_phase0_frame = idx + 1  # convert to 1-based
                        break

                if early_phase0_frame is not None and early_phase0_frame < nxt_start:
                    # Clip attach to just before the phase0 hit
                    cur["end_frame"] = max(cur_start, early_phase0_frame - 1)

                    # Re-match the take template from the early phase0 frame
                    rematch_idx = early_phase0_frame - 1  # 0-based
                    rematch_window = frames[
                        rematch_idx : rematch_idx + self.max_window_frames
                    ]
                    start_rel, end_rel, conf, timer_rel, _ = nxt_t.try_match(
                        rematch_window, inherited_groups
                    )

                    if end_rel > 0 and conf > 0:
                        new_semantic = (
                            rematch_idx + start_rel + 1
                            if start_rel >= 0
                            else early_phase0_frame
                        )
                        new_end = min(len(frames), rematch_idx + end_rel)
                        # Guard: do not extend past the action after nxt
                        if i + 2 < len(out):
                            next_next_start = self._ordering_start(out[i + 2])
                            new_end = min(new_end, next_next_start - 1)
                        nxt["start_frame"] = early_phase0_frame
                        nxt["match_start_frame"] = early_phase0_frame
                        nxt["semantic_start_frame"] = new_semantic
                        nxt["end_frame"] = new_end
                        nxt["confidence"] = round(conf, 4)
                    # else: clip attach but keep original take boundaries
                    continue

            clipped_end = min(cur_end, nxt_start - 1)
            cur["end_frame"] = max(cur_start, clipped_end)
        return out

    def _refine_take_timer_by_hand(
        self,
        sequence: List[Dict[str, Any]],
        frames: List[List[Detection]],
        inherited_groups: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """Refine take timer using only the hand(s) that actually enter the container.
        This prevents false-positive timer starts from an idle hand that was already
        near the container before the actual take started."""
        out: List[Dict[str, Any]] = []
        for item in sequence:
            tmpl: ActionTemplate = item["template"]
            if self._template_role(tmpl) != "take":
                out.append(item)
                continue

            candidate_hand_ids = item.get("hand_track_ids")
            if not candidate_hand_ids:
                out.append(item)
                continue

            # Identify which hand(s) actually entered a Container during the take.
            take_hand_ids = self._hands_in_container(
                frames, inherited_groups, item, candidate_hand_ids
            )
            if not take_hand_ids:
                # No hand entered a container – keep as is (backward compat).
                out.append(item)
                continue

            # Re-find phase 0 using ONLY the hand(s) that entered the container.
            refined = tmpl.find_first_phase0_with_hands(
                frames,
                inherited_groups,
                take_hand_ids,
                int(item["start_frame"]) - 1,
                int(item["end_frame"]),
            )
            if refined is not None:
                item["timer_start_frame"] = refined + 1
                out.append(item)
            else:
                # Phase 0 not met by any hand that entered container -> drop
                continue
        return out

    def _refine_attach_timer_by_hand(
        self,
        sequence: List[Dict[str, Any]],
        frames: List[List[Detection]],
        inherited_groups: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """Refine attach timer using the same hand that performed the preceding take.
        Attach candidates whose phase 0 cannot be satisfied by the take-hand are
        removed from the sequence (false positive caused by the other hand idling
        near the target)."""
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(sequence):
            tmpl: ActionTemplate = item["template"]
            if tmpl.base_id not in ("attach_component", "attach_only"):
                out.append(item)
                continue

            attach_start = int(item["start_frame"])
            prev_take_hand_ids: Optional[set] = None
            # Search backwards through *already-validated* output to find the
            # most recent take that ends before this attach.
            for j in range(len(out) - 1, -1, -1):
                prev = out[j]
                prev_tmpl: ActionTemplate = prev["template"]
                if self._template_role(prev_tmpl) == "take" and int(prev["end_frame"]) < attach_start:
                    prev_take_hand_ids = prev.get("hand_track_ids")
                    break

            if not prev_take_hand_ids:
                # No prior take to anchor to – keep the attach (backward compat).
                out.append(item)
                continue

            # Identify which hand(s) actually entered a Container during the take.
            take_hand_ids = self._hands_in_container(
                frames, inherited_groups, prev, prev_take_hand_ids
            )
            if not take_hand_ids:
                # Take hand never entered a container – keep attach (backward compat).
                out.append(item)
                continue

            # First, try phase 0 with only the container-entering hand(s).
            refined = tmpl.find_first_phase0_with_hands(
                frames,
                inherited_groups,
                take_hand_ids,
                int(item["start_frame"]) - 1,
                int(item["end_frame"]),
            )
            if refined is not None:
                item["timer_start_frame"] = refined + 1
                out.append(item)
                continue

            # The take-hand is not in the target area.  Try with *all* hands from
            # the preceding take – the worker may have switched hands for the attach.
            refined = tmpl.find_first_phase0_with_hands(
                frames,
                inherited_groups,
                prev_take_hand_ids,
                int(item["start_frame"]) - 1,
                int(item["end_frame"]),
            )
            if refined is None:
                print(f"[DEBUG] Dropping {tmpl.base_id} because Phase 0 not met by hand(s) {prev_take_hand_ids}")
                continue  # phase 0 not met by any hand → drop

            # Phase 0 is met by a non-take hand.  If the take-hand is *still* inside
            # a Container at the attach start, the "attach" is just the other hand
            # idling near the target while the worker is still taking → drop it.
            if self._is_hand_in_container_at(
                frames, inherited_groups, take_hand_ids, int(item["start_frame"])
            ):
                continue

            item["timer_start_frame"] = refined + 1
            out.append(item)
        return out

    @staticmethod
    def _hands_in_container(
        frames: List[List[Detection]],
        inherited_groups: Dict[str, List[str]],
        take_item: Dict[str, Any],
        candidate_hand_ids: set,
    ) -> set:
        """Return the subset of *candidate_hand_ids* whose bounding-box centre
        falls inside any Container detection during the take item's timespan."""
        container_classes = set(inherited_groups.get("Container", []))
        if not container_classes:
            return candidate_hand_ids  # no container definition → keep all

        start_f = max(0, int(take_item.get("timer_start_frame", take_item.get("semantic_start_frame", take_item["start_frame"]))) - 1)
        end_f = min(len(frames), int(take_item["end_frame"]))
        active: set = set()
        for fi in range(start_f, end_f):
            containers = [d for d in frames[fi] if d.cls in container_classes]
            if not containers:
                continue
            for d in frames[fi]:
                if d.cls != "hand" or d.track_id is None:
                    continue
                if d.track_id not in candidate_hand_ids:
                    continue
                # Simple centre-point-in-bbox test for each container.
                cx = sum(p[0] for p in d.poly) / len(d.poly)
                cy = sum(p[1] for p in d.poly) / len(d.poly)
                for c in containers:
                    cxs = [p[0] for p in c.poly]
                    cys = [p[1] for p in c.poly]
                    if min(cxs) <= cx <= max(cxs) and min(cys) <= cy <= max(cys):
                        active.add(d.track_id)
                        break
        return active if active else candidate_hand_ids

    @staticmethod
    def _is_hand_in_container_at(
        frames: List[List[Detection]],
        inherited_groups: Dict[str, List[str]],
        hand_track_ids: set,
        frame_idx: int,
    ) -> bool:
        """Return True if any hand in *hand_track_ids* has its centre inside
        a Container detection at the given frame index."""
        container_classes = set(inherited_groups.get("Container", []))
        if not container_classes or frame_idx < 0 or frame_idx >= len(frames):
            return False
        containers = [d for d in frames[frame_idx] if d.cls in container_classes]
        if not containers:
            return False
        for d in frames[frame_idx]:
            if d.cls != "hand" or d.track_id not in hand_track_ids:
                continue
            cx = sum(p[0] for p in d.poly) / len(d.poly)
            cy = sum(p[1] for p in d.poly) / len(d.poly)
            for c in containers:
                cxs = [p[0] for p in c.poly]
                cys = [p[1] for p in c.poly]
                if min(cxs) <= cx <= max(cxs) and min(cys) <= cy <= max(cys):
                    return True
        return False

    def infer_from_detections(self, detections_jsonl: Path) -> Dict[str, Any]:
        frames = load_detection_frames(detections_jsonl, min_conf=self.min_det_conf)
        if not frames:
            raise ValueError(f"No valid detections in: {detections_jsonl}")

        inherited_groups = self.global_cfg.get("class_groups", {})
        candidates = self._collect_candidates(frames, inherited_groups)
        matched_sequence = self._decode_best_path_stateful(candidates)
        matched_sequence = self._augment_intermediate_candidates(matched_sequence, candidates)
        matched_sequence = self._attach_return_candidate(matched_sequence, candidates)
        matched_sequence = self._trim_generic_attach_before_next_take(
            matched_sequence, frames, inherited_groups
        )
        matched_sequence = self._refine_take_timer_by_hand(
            matched_sequence, frames, inherited_groups
        )
        matched_sequence = self._refine_attach_timer_by_hand(
            matched_sequence, frames, inherited_groups
        )

        if not matched_sequence:
            raise ValueError(
                "No action matched above threshold. Lower --conf-threshold or verify detections/templates."
            )
        return self._build_yaml(matched_sequence)

    @staticmethod
    def _sanitize_component_name(name: str) -> str:
        """Replace spaces with underscores for safe action IDs."""
        return name.replace(" ", "_")

    def _build_yaml(self, sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Ensure temporal order so pending_take_info accumulates correctly.
        # Use _ordering_start (actual action start) not start_frame (window scan start).
        sequence = sorted(sequence, key=lambda x: self._ordering_start(x))
        actions_out: List[Dict[str, Any]] = []
        repeat_counters: Dict[str, int] = {}
        matches_meta: List[Dict[str, Any]] = []
        take_component_count = 0
        pending_take_info: List[Tuple[int, str]] = []  # (count, component_name)
        last_component_name = "component"
        used_action_ids: set = set()

        for item in sequence:
            tmpl: ActionTemplate = item["template"]
            base_id = tmpl.base_id
            repeat_counters[base_id] = repeat_counters.get(base_id, 0) + 1
            count = repeat_counters[base_id]
            role = self._template_role(tmpl)

            action = deepcopy(tmpl.action)
            default_id = str(action.get("id", base_id))
            final_id = default_id
            final_count_suffix: Optional[str] = None

            if role == "take" and self._is_generic_take_base_id(base_id):
                take_component_count += 1
                comp_name = item.get("detected_component") or "component"
                comp_name = self._sanitize_component_name(comp_name)
                last_component_name = comp_name
                pending_take_info.append((take_component_count, comp_name))
                final_id = f"take_{comp_name}_{take_component_count}"
                final_count_suffix = str(take_component_count)
            elif role == "attach" and base_id in ("attach_component", "attach_only"):
                if pending_take_info:
                    if len(pending_take_info) == 1:
                        cnt, comp_name = pending_take_info[0]
                        suffix = str(cnt)
                    else:
                        cnts = [str(c) for c, _ in pending_take_info]
                        comps = [cn for _, cn in pending_take_info]
                        suffix = "_and_".join(cnts)
                        comp_name = "_and_".join(comps)
                    final_id = f"attach_{comp_name}_{suffix}"
                    final_count_suffix = suffix
                    pending_take_info = []
                elif count > 1 or tmpl.repeatable:
                    comp_name = last_component_name
                    final_id = f"attach_{comp_name}_{count}"
                    final_count_suffix = str(count)
            elif count > 1 or tmpl.repeatable:
                final_id = f"{base_id}_{count}"
                final_count_suffix = str(count)

            action["id"] = final_id
            if final_id in used_action_ids:
                final_id = f"{final_id}_r{count}"
                action["id"] = final_id
            used_action_ids.add(final_id)
            if final_count_suffix:
                desc = str(action.get("description", "")).strip()
                action["description"] = f"{desc} (lần {final_count_suffix})".strip()

            actions_out.append(action)
            eff_start = int(item.get("timer_start_frame", item.get("semantic_start_frame", item["start_frame"])))
            if matches_meta:
                prev_end = int(matches_meta[-1]["end_frame"])
                candidate_start = max(eff_start, prev_end + 1)
                if candidate_start <= int(item["end_frame"]):
                    eff_start = candidate_start
            eff_start = min(eff_start, int(item["end_frame"]))
            matches_meta.append(
                {
                    "action_id": action["id"],
                    "source_template": str(tmpl.path),
                    "start_frame": eff_start,
                    "match_start_frame": int(item["start_frame"]),
                    "end_frame": item["end_frame"],
                    "confidence": item["confidence"],
                }
            )

        merged_global = deepcopy(self.global_cfg)
        merged_global["handoff_prev_end_to_next_start"] = bool(
            merged_global.get("handoff_prev_end_to_next_start", True)
        )
        return {
            "global": merged_global,
            "actions": actions_out,
            "_inference_meta": {
                "total_matches": len(matches_meta),
                "matches": matches_meta,
            },
        }

    def build_timeline_rows(self, inferred: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for idx, m in enumerate(inferred.get("_inference_meta", {}).get("matches", []), start=1):
            rows.append(
                {
                    "order": idx,
                    "action_id": str(m.get("action_id", "")),
                    "description": "",
                    "start_frame": int(m.get("start_frame", -1)),
                    "end_frame": int(m.get("end_frame", -1)),
                    "clip_start_frame": None,
                    "clip_end_frame": None,
                    "outcome": "inferred",
                    "end_reason": "dynamic_sequence_decode",
                    "clip_path": "",
                }
            )
        return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: infer ordered SOP from template actions and detection stream."
    )
    parser.add_argument(
        "--templates",
        type=str,
        nargs="+",
        help="Template YAML files, each should contain exactly one action.",
    )
    parser.add_argument(
        "--detections",
        type=str,
        help="Detections JSONL path produced by inference_camera.py --detections.",
    )
    parser.add_argument(
        "--global-config",
        type=str,
        default="",
        help="Optional shared global YAML (class_groups, handoff...). Recommended for reuse.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output inferred SOP YAML path.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=DEFAULT_TUNING["conf_threshold"],
        help="Minimum confidence to accept a match. Default: 0.55",
    )
    parser.add_argument(
        "--max-window-frames",
        type=int,
        default=DEFAULT_TUNING["max_window_frames"],
        help="Frame window used for matching each action attempt. Default: 300",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Optional video path to render timer overlay output.",
    )
    parser.add_argument(
        "--overlay-output",
        type=str,
        default="",
        help="Optional overlay mp4 path. Default: <output_stem>_timer_overlay.mp4",
    )
    parser.add_argument(
        "--json-timeline",
        type=str,
        default="",
        help="Path to a JSON file containing timeline rows to render directly.",
    )
    return parser.parse_args()


def render_video_with_boxes_and_timer(
    video_path: Path,
    out_path: Path,
    detections_jsonl: Path,
    timeline_rows: List[dict],
    min_conf: float = 0.0,
) -> None:
    frames_dets = load_detection_frames(detections_jsonl, min_conf=min_conf)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps_in <= 0:
        fps_in = 30.0
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

        frame_idx += 1

        dets = frames_dets[frame_idx - 1] if (frame_idx - 1) < len(frames_dets) else []
        frame = draw_detection_boxes(frame, dets)

        while seg_idx < len(segments) and frame_idx > segments[seg_idx][1]:
            seg_idx += 1
        cur = segments[seg_idx] if seg_idx < len(segments) else (frame_idx, frame_idx, "idle")
        cur_s, cur_e, cur_id = int(cur[0]), int(cur[1]), str(cur[2])

        last_id = str(rows[-1]["action_id"]) if rows else ""
        if not rows or last_id != cur_id:
            if rows:
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

    writer.release()
    cap.release()


def main() -> None:
    args = parse_args()
    
    if args.json_timeline:
        timeline_rows = json.loads(Path(args.json_timeline).read_text(encoding="utf-8"))
        video_path = Path(args.video)
        overlay_path = Path(args.overlay_output)
        detections_path = Path(args.detections) if args.detections else None
        render_video_with_boxes_and_timer(video_path, overlay_path, detections_path, timeline_rows, min_conf=0.25)
        print(f"✅ JSON-based timer overlay video written: {overlay_path}")
        return

    # Normal mode requires these
    if not args.templates or not args.detections or not args.output:
        print("Error: --templates, --detections and --output are required unless --json-timeline is provided.")
        return

    template_paths = [Path(p) for p in args.templates]
    detections_path = Path(args.detections)
    output_path = Path(args.output)

    missing = [str(p) for p in (template_paths + [detections_path]) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing file(s): {missing}")

    global_cfg: Dict[str, Any] = {}
    if args.global_config:
        gpath = Path(args.global_config)
        if not gpath.exists():
            raise FileNotFoundError(f"Global config not found: {gpath}")
        global_cfg = load_yaml(gpath).get("global", {})

    builder = DynamicSOPBuilder(
        template_paths=template_paths,
        global_cfg=global_cfg,
        conf_threshold=args.conf_threshold,
        max_window_frames=args.max_window_frames,
    )
    inferred = builder.infer_from_detections(detections_path)

    output_path.write_text(
        yaml.safe_dump(inferred, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"✅ Inferred SOP written: {output_path}")
    print(f"   Actions: {len(inferred.get('actions', []))}")

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found for overlay render: {video_path}")

        overlay_path = (
            Path(args.overlay_output)
            if args.overlay_output
            else output_path.with_name(f"{output_path.stem}_timer_overlay.mp4")
        )
        timeline_rows = builder.build_timeline_rows(inferred)
        render_video_with_boxes_and_timer(video_path, overlay_path, detections_path, timeline_rows, min_conf=0.25)
        print(f"✅ Timer overlay video written: {overlay_path}")


if __name__ == "__main__":
    main()
