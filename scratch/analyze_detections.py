import json
from collections import defaultdict
import numpy as np

files = {
    "xb10_2_8": "xb10_2_8_detections.jsonl",
    "xb10_1_13": "xb10_1_13_detections.jsonl",
}

for label, path in files.items():
    class_confs = defaultdict(list)
    class_frames = defaultdict(set)
    total_frames = 0

    with open(path, 'r') as f:
        for line in f:
            row = json.loads(line.strip())
            frame = row.get('frame', 0)
            total_frames += 1
            for d in row.get('detections', []):
                cls = d.get('class', d.get('cls', 'unknown'))
                conf = d.get('conf', 0)
                class_confs[cls].append(conf)
                class_frames[cls].add(frame)

    print(f"\n{'='*90}")
    print(f"  {label} ({path}) - Total frames: {total_frames}")
    print(f"{'='*90}")
    header = f"{'Class':<15} {'Count':>6} {'Frames':>7} {'Coverage':>8} {'ConfMin':>8} {'ConfMed':>8} {'ConfMean':>8} {'ConfMax':>8}"
    print(header)
    print('-' * 85)
    for cls in sorted(class_confs.keys(), key=lambda x: -len(class_confs[x])):
        confs = class_confs[cls]
        frames = class_frames[cls]
        coverage = f"{len(frames) / total_frames * 100:.1f}%"
        arr = np.array(confs)
        print(f"{cls:<15} {len(confs):>6} {len(frames):>7} {coverage:>8} {arr.min():>8.3f} {np.median(arr):>8.3f} {arr.mean():>8.3f} {arr.max():>8.3f}")

# Deep-dive: bracket detection in xb10_2_8
print(f"\n\n{'='*90}")
print("  BRACKET analysis in xb10_2_8 - checking spatial relationship with board/gasket")
print(f"{'='*90}")

with open("xb10_2_8_detections.jsonl", 'r') as f:
    bracket_with_board = 0
    bracket_without_board = 0
    bracket_total = 0
    gasket_with_bracket = 0
    gasket_total_frames = 0
    
    for line in f:
        row = json.loads(line.strip())
        dets = row.get('detections', [])
        classes_in_frame = [d.get('class', '') for d in dets]
        
        has_bracket = 'bracket' in classes_in_frame
        has_board = 'board' in classes_in_frame
        has_gasket = 'gasket' in classes_in_frame
        
        if has_bracket:
            bracket_total += 1
            if has_board:
                bracket_with_board += 1
            else:
                bracket_without_board += 1
        
        if has_gasket:
            gasket_total_frames += 1
            if has_bracket:
                gasket_with_bracket += 1

    print(f"Bracket appears in {bracket_total} frames")
    print(f"  - With board present: {bracket_with_board}")
    print(f"  - Without board: {bracket_without_board}")
    print(f"Gasket appears in {gasket_total_frames} frames")
    print(f"  - With bracket co-present: {gasket_with_bracket}")

# Check temporal distribution of key classes
print(f"\n\n{'='*90}")
print("  TEMPORAL GAPS in xb10_2_8 - showing frame ranges where board is present")
print(f"{'='*90}")

board_frames = []
gasket_frames = []
bracket_frames = []
hand_in_tray_frames = []

with open("xb10_2_8_detections.jsonl", 'r') as f:
    for line in f:
        row = json.loads(line.strip())
        frame = row.get('frame', 0)
        dets = row.get('detections', [])
        classes_in_frame = {d.get('class', '') for d in dets}
        
        if 'board' in classes_in_frame:
            board_frames.append(frame)
        if 'gasket' in classes_in_frame:
            gasket_frames.append(frame)
        if 'bracket' in classes_in_frame:
            bracket_frames.append(frame)
        if 'hand' in classes_in_frame and 'tray' in classes_in_frame:
            hand_in_tray_frames.append(frame)

def find_segments(frames):
    if not frames:
        return []
    segments = []
    start = frames[0]
    prev = frames[0]
    for f in frames[1:]:
        if f > prev + 5:
            segments.append((start, prev))
            start = f
        prev = f
    segments.append((start, prev))
    return segments

print(f"\nBoard presence segments:")
for s, e in find_segments(board_frames):
    print(f"  frames {s}-{e} ({e-s+1} frames, ~{(e-s+1)/30:.1f}s)")

print(f"\nGasket presence segments:")
for s, e in find_segments(gasket_frames):
    print(f"  frames {s}-{e} ({e-s+1} frames, ~{(e-s+1)/30:.1f}s)")

print(f"\nBracket presence segments:")
for s, e in find_segments(bracket_frames):
    print(f"  frames {s}-{e} ({e-s+1} frames, ~{(e-s+1)/30:.1f}s)")

print(f"\nHand + Tray co-present segments (potential take actions):")
for s, e in find_segments(hand_in_tray_frames):
    dur = e - s + 1
    if dur >= 5:  # only show meaningful segments
        print(f"  frames {s}-{e} ({dur} frames, ~{dur/30:.1f}s)")
