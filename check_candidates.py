import yaml
from pathlib import Path
from dynamic_sop_builder import DynamicSOPBuilder, load_detection_frames

def check_candidates():
    global_cfg = yaml.safe_load(Path("sop_global_shared.yaml").read_text())
    builder = DynamicSOPBuilder(
        template_paths=[Path("attach_screw.yaml")],
        global_cfg=global_cfg
    )
    frames = load_detection_frames(Path("xb10_5_3_detections.jsonl"))
    
    # We only care about attach_screw template
    tmpl = builder.templates[0]
    
    candidates = []
    for start in range(0, len(frames)):
        start_rel, end_rel, conf, timer_rel, hand_ids = tmpl.try_match(
            frames[start : start + 300], {}
        )
        if end_rel > 0 and conf >= 0.5:
            candidates.append({
                "start": start + 1,
                "end": start + end_rel,
                "timer": start + timer_rel + 1,
                "conf": conf
            })
    
    for c in sorted(candidates, key=lambda x: x["start"]):
        print(f"Cand: {c['start']}-{c['end']} (timer {c['timer']}) conf={c['conf']:.2f}")

if __name__ == "__main__":
    check_candidates()
