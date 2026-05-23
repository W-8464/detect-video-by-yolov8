import yaml
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from segment_actions_refactored import Detection, evaluate_condition

class StreamingActionEngine:
    """
    Action Engine optimized for Real-time streaming.
    Uses time-based logic (seconds) instead of raw frame counts
    to handle variable FPS environments.
    Supports multiple config files - actions are merged in order,
    global settings are taken from the first file.
    """
    def __init__(self, config_path, reference_fps: float = 30.0):
        # Support both single str and list of str paths
        if isinstance(config_path, str):
            config_paths = [config_path]
        else:
            config_paths = list(config_path)

        # Load first config for global settings
        first_cfg = yaml.safe_load(Path(config_paths[0]).read_text(encoding="utf-8"))
        g = first_cfg.get("global", {})
        self.handoff = bool(g.get("handoff_prev_end_to_next_start", False))
        self.class_groups = g.get("class_groups", {})
        self.reference_fps = reference_fps

        # Merge actions from all configs in order
        merged_actions = []
        for p in config_paths:
            cfg = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
            # Also merge class_groups from all files (later files can add more groups)
            extra_groups = cfg.get("global", {}).get("class_groups", {})
            for k, v in extra_groups.items():
                if k not in self.class_groups:
                    self.class_groups[k] = v
            merged_actions.extend(cfg.get("actions", []))

        self.actions = merged_actions
        self.cfg = first_cfg  # kept for reference
        
        self.current_action_idx = 0
        self.current_phase_idx = 0
        
        # Time-based tracking
        self.streak_start_time: Optional[float] = None
        self.miss_start_time: Optional[float] = None
        
        # Look-ahead tracking
        self.lookahead_streak_start_time: Optional[float] = None
        
        # Action-level timing
        self.action_start_time: Optional[float] = None
        self.done = False
        self.history = []

    def reset(self):
        """Reset engine to process a new sequence starting from action 0."""
        self.current_action_idx = 0
        self.current_phase_idx = 0
        self.streak_start_time = None
        self.miss_start_time = None
        self.lookahead_streak_start_time = None
        self.action_start_time = None
        self.done = False
        self.history = []

    def process_frame(self, frame_idx: int, dets: List[Detection]) -> Dict[str, Any]:
        """
        Process a single frame. Uses system time for state transitions.
        """
        now = time.time()
        
        if self.done or self.current_action_idx >= len(self.actions):
            return self._current_state(frame_idx)
            
        action = self.actions[self.current_action_idx]
        act_id = action.get("id", f"action_{self.current_action_idx+1}")
        phases = action.get("phases", [])
        
        # Convert max_action_frames to seconds
        max_action_frames = int(action.get("max_action_frames", 0))
        max_action_secs = max_action_frames / self.reference_fps if max_action_frames > 0 else 0
        
        force_start = bool(action.get("force_start", False))
        
        if force_start and self.action_start_time is None:
            self.action_start_time = now

        # Check Timeout (restored)
        if self.action_start_time is not None and max_action_secs > 0:
            if (now - self.action_start_time) >= max_action_secs:
                print(f"[DEBUG Engine] ⏰ Timeout reached for '{act_id}'. Force closing.")
                self._complete_current_action(now, act_id, "completed", "max_action_frames_cap")
                return self._current_state(frame_idx)

        # Check if we already finished all phases (edge case)
        if self.current_phase_idx >= len(phases):
            self._complete_current_action(now, act_id, "completed", "naturally_ended")
            return self._current_state(frame_idx)
            
        phase = phases[self.current_phase_idx]
        conds = phase.get("conditions", [])
        
        # Convert frame thresholds to seconds
        hold_frames = int(phase.get("hold_frames", 1))
        grace_miss_frames = int(phase.get("grace_miss", 2))
        
        hold_secs = hold_frames / self.reference_fps
        grace_secs = grace_miss_frames / self.reference_fps
        
        # Evaluate current phase
        phase_met = True
        if conds:
            phase_met = all(evaluate_condition(dets, c, self.class_groups) for c in conds)
            
        if phase_met:
            # We are currently in a "success" state
            if self.streak_start_time is None:
                self.streak_start_time = now
                print(f"[DEBUG Engine] {act_id} Phase {self.current_phase_idx} ({phase.get('id')}) STREAK STARTED")
            
            # Reset miss timer
            self.miss_start_time = None
            self.lookahead_streak_start_time = None  # Cancel any look-ahead since we are succeeding
            
            # Start action timer on configured phase (default 0)
            timer_start_phase = int(action.get("timer_start_phase", 0))
            if self.action_start_time is None and self.current_phase_idx == timer_start_phase:
                self.action_start_time = now
                print(f"[DEBUG Engine] {act_id} ACTION STARTED (Timer) at {now}")
                
            # Check if hold time reached
            if self.streak_start_time is not None and (now - self.streak_start_time) >= hold_secs:
                print(f"[DEBUG Engine] ✔️ {act_id} Phase {self.current_phase_idx} ({phase.get('id')}) MET HOLD SECS = {hold_secs:.2f}")
                self.current_phase_idx += 1
                self.streak_start_time = None
                self.miss_start_time = None
                
                # Check if action completed
                if self.current_phase_idx >= len(phases):
                    self._complete_current_action(now, act_id, "completed", "naturally_ended")
        else:
            # Condition not met - Current action is "stuck" or "waiting"
            if self.streak_start_time is not None:
                print(f"[DEBUG Engine] ❌ {act_id} Phase {self.current_phase_idx} ({phase.get('id')}) STREAK BROKEN")
                self.streak_start_time = None
            
            if self.action_start_time is not None:
                if self.miss_start_time is None:
                    self.miss_start_time = now
                
                # Check if grace period exceeded
                if self.miss_start_time is not None and (now - self.miss_start_time) > grace_secs:
                    self.miss_start_time = None
                    timer_start_phase = int(action.get("timer_start_phase", 0))
                    if self.current_phase_idx == timer_start_phase and not force_start:
                        self.action_start_time = None
                        print(f"[DEBUG Engine] ↩️ {act_id} ACTION RE-SET (lost at timer start phase limits)")
            
            # -----------------------------------------------------
            # LOOK-AHEAD: Handoff if next action's Phase 0 is met
            # Only evaluate this if the current action is NOT making progress
            # -----------------------------------------------------
            if self.handoff and self.current_action_idx + 1 < len(self.actions):
                next_action = self.actions[self.current_action_idx + 1]
                if next_action.get("allow_lookahead", False) and not next_action.get("force_start", False):
                    next_phases = next_action.get("phases", [])
                    if next_phases:
                        next_p0 = next_phases[0]
                        next_conds = next_p0.get("conditions", [])
                        if all(evaluate_condition(dets, c, self.class_groups) for c in next_conds):
                            if self.lookahead_streak_start_time is None:
                                self.lookahead_streak_start_time = now
                                print(f"[DEBUG Engine] Look-ahead started for next action: {next_action.get('id')}")
                            
                            next_hold_secs = int(next_p0.get("hold_frames", 1)) / self.reference_fps
                            if (now - self.lookahead_streak_start_time) >= next_hold_secs:
                                print(f"[DEBUG Engine] 🚨 HANDOFF TRIGGERED! Force closing '{act_id}' & jumping to '{next_action.get('id')}'")
                                self._complete_current_action(now, act_id, "pending_handoff", "handoff_prev_end_to_next_start")
                                
                                # Handoff occurred because Next Action's Phase 0 was met and held.
                                # Therefore we physically consider Phase 0 completed for Next Action!
                                self.current_phase_idx = 1
                                self.streak_start_time = None
                                self.miss_start_time = None
                                self.lookahead_streak_start_time = None
                                return self._current_state(frame_idx)
                        else:
                            if self.lookahead_streak_start_time is not None:
                                print(f"[DEBUG Engine] Look-ahead broken for next action: {next_action.get('id')}")
                                self.lookahead_streak_start_time = None
                        
        return self._current_state(frame_idx)

    def _complete_current_action(self, end_time: float, act_id: str, outcome: str, end_reason: str):
        self.history.append({
            "action_id": act_id,
            "start_time": self.action_start_time,
            "end_time": end_time,
            "outcome": outcome,
            "end_reason": end_reason
        })
        self.current_action_idx += 1
        self.current_phase_idx = 0
        self.streak_start_time = None
        self.miss_start_time = None
        self.lookahead_streak_start_time = None
        self.action_start_time = None
        
        # Handoff logic
        if self.handoff and self.current_action_idx < len(self.actions):
            next_action = self.actions[self.current_action_idx]
            if not bool(next_action.get("force_start", False)):
                self.action_start_time = end_time

        if self.current_action_idx >= len(self.actions):
            self.done = True

    def _current_state(self, current_frame_idx: int) -> Dict[str, Any]:
        now = time.time()
        if self.done:
            return {"status": "done", "action_id": "All Completed!", "history": self.history}
            
        action = self.actions[self.current_action_idx]
        act_id = action.get("id", str(self.current_action_idx))
        
        return {
            "status": "running",
            "action_idx": self.current_action_idx,
            "total_actions": len(self.actions),
            "action_id": act_id,
            "phase_idx": self.current_phase_idx,
            "total_phases": len(action.get("phases", [])),
            "start_time": self.action_start_time,
            "now": now,
            "history": self.history
        }
