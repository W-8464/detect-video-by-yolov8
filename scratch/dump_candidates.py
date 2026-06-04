import argparse
from pathlib import Path
import json
import yaml
from dynamic_sop_builder import DynamicSOPBuilder, ActionTemplate

def dump_candidates(video_name, detections_path, config_path, templates):
    templates = [Path(t) for t in templates]
    global_cfg = yaml.safe_load(Path(config_path).read_text())["global"]
    builder = DynamicSOPBuilder(templates, global_cfg=global_cfg)
    
    from dynamic_sop_builder import load_detection_frames
    frames = load_detection_frames(Path(detections_path), min_conf=builder.min_det_conf)
    inherited_groups = builder.global_cfg.get("class_groups", {})
    candidates = builder._collect_candidates(frames, inherited_groups)
    
    output = []
    for c in candidates:
        output.append({
            "template": str(c["template"].path.name),
            "start": c["start_frame"],
            "end": c["end_frame"],
            "conf": c["confidence"],
            "comp": c["detected_component"]
        })
    
    with open(f"candidates_{video_name}.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--detections")
    parser.add_argument("--config")
    parser.add_argument("--templates", nargs="+")
    args = parser.parse_args()
    dump_candidates(args.name, args.detections, args.config, args.templates)
