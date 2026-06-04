import argparse
from pathlib import Path
import json
import yaml
from dynamic_sop_builder import DynamicSOPBuilder

def compare_runs(detections_path, config_path, templates, limit_frames=None):
    templates = [Path(t) for t in templates]
    global_cfg = yaml.safe_load(Path(config_path).read_text())["global"]
    builder = DynamicSOPBuilder(templates, global_cfg=global_cfg)
    
    from dynamic_sop_builder import load_detection_frames
    frames = load_detection_frames(Path(detections_path), min_conf=builder.min_det_conf)
    if limit_frames:
        frames = frames[:limit_frames]
        
    inherited_groups = builder.global_cfg.get("class_groups", {})
    candidates = builder._collect_candidates(frames, inherited_groups)
    matched_sequence = builder._decode_best_path_stateful(candidates)
    
    output = []
    for item in matched_sequence:
        output.append({
            "id": item.get("action_id", str(item["template"].path.name)),
            "start": item["start_frame"],
            "end": item["end_frame"],
            "conf": item["confidence"]
        })
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections")
    parser.add_argument("--config")
    parser.add_argument("--templates", nargs="+")
    args = parser.parse_args()
    
    # Run 1: Full detections
    path_full = compare_runs(args.detections, args.config, args.templates)
    
    # Run 2: Cut detections (first 740 frames)
    path_cut = compare_runs(args.detections, args.config, args.templates, limit_frames=740)
    
    print("--- FULL RUN (First 10 actions) ---")
    for i, a in enumerate(path_full[:10]):
        print(f"{i}: {a['id']} ({a['start']}-{a['end']})")
        
    print("\n--- CUT RUN (First 10 actions) ---")
    for i, a in enumerate(path_cut[:10]):
        print(f"{i}: {a['id']} ({a['start']}-{a['end']})")
