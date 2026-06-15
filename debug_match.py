import yaml
from pathlib import Path
from segment_actions_refactored import evaluate_condition
from dynamic_sop_builder import load_detection_frames, ActionTemplate

def debug_match(tmpl_path, detections_path, frame_start, frame_end):
    global_cfg = yaml.safe_load(Path("sop_global_shared.yaml").read_text())
    tmpl = ActionTemplate(Path(tmpl_path), tuning=global_cfg.get("tuning", {}))
    frames = load_detection_frames(Path(detections_path))
    
    phases = tmpl.action.get("phases", [])
    class_groups = global_cfg.get("class_groups", {})
    
    for start_frame in range(frame_start, frame_end):
        window = frames[start_frame : start_frame + 300]
        if not window: continue
        
        cursor = 0
        actual_start = -1
        
        for p_idx, phase in enumerate(phases):
            hold = int(phase.get("hold_frames", 1))
            grace = int(phase.get("grace_miss", 0))
            matched = 0
            missed = 0
            
            p_start = cursor
            found_phase = False
            while cursor < len(window):
                conds = phase.get("conditions", [])
                curr_f = window[cursor]
                hist_f = window[max(0, cursor-30):cursor]
                
                met = all(evaluate_condition(curr_f, c, class_groups, hist_f) for c in conds)
                
                if met:
                    matched += 1
                    missed = 0
                    if matched == 1 and actual_start == -1:
                        actual_start = start_frame + cursor
                    if matched >= hold:
                        cursor += 1
                        found_phase = True
                        break
                else:
                    if matched == 0:
                        if cursor < 220: # phase_start_wait_frames
                            cursor += 1
                            continue
                        else: break
                    else:
                        missed += 1
                        if missed > grace: break
                cursor += 1
            
            if not found_phase:
                if p_idx >= 0 and matched > 0:
                    # print(f"Start {start_frame}: Failed at Phase {p_idx} (matched {matched}/{hold})")
                    pass
                break
        else:
            print(f"MATCH FOUND starting around {start_frame}!! Actual start: {actual_start}, End: {start_frame + cursor}")

if __name__ == "__main__":
    # Check gap between 3 and 4
    print("Checking Gap 503-573...")
    debug_match("attach_screw.yaml", "xb10_5_3_detections.jsonl", 500, 560)
    # Check gap after 4
    print("Checking Gap 634-742...")
    debug_match("attach_screw.yaml", "xb10_5_3_detections.jsonl", 630, 720)
