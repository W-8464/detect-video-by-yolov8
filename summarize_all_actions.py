import csv
import os

THRES_PCIE = 0.02
THRES_SHIELD = 0.008

IDEAL_PCIE = 0.12
IDEAL_SHIELD = 0.08

def calc_confidence(action_id, best_score, outcome):
    best_score = float(best_score)
    if best_score == -1.0 or outcome == "blocked":
        return 0.0
    
    if action_id in ["board_to_jig", "board_back_to_conveyor"]:
        return min(100.0, best_score * 100)
        
    score = 0.0
    if action_id == "pcie_from_tray_to_board":
        if best_score < THRES_PCIE:
            score = 30.0
        else:
            score = 50.0 + 50.0 * (best_score - THRES_PCIE) / (IDEAL_PCIE - THRES_PCIE)
    elif action_id == "shielding_from_tray":
        if best_score < THRES_SHIELD:
            score = 30.0
        else:
            score = 50.0 + 50.0 * (best_score - THRES_SHIELD) / (IDEAL_SHIELD - THRES_SHIELD)
            
    score = min(100.0, max(0.0, score))
    
    if outcome == "pending_handoff":
        score = max(0.0, score - 30.0) # Penalty
        
    return round(score, 1)

output_csv = "all_videos_summary.csv"
videos = ["xb10_6", "xb10_7", "xb10_8", "xb10_9", "xb10_10"]
all_rows = []

headers = ["Video", "Action", "Start", "End", "Outcome", "Start Reason", "End Reason", "Raw Score", "Confidence (/100)"]

for v in videos:
    path = f"runs/actions_{v}_with_reasons/timeline.csv"
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        continue
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conf = calc_confidence(row['action_id'], row['best_score'], row['outcome'])
            all_rows.append({
                "Video": v,
                "Action": row['action_id'],
                "Start": row['start_frame'],
                "End": row['end_frame'],
                "Outcome": row['outcome'],
                "Start Reason": row['start_reason'],
                "End Reason": row['end_reason'],
                "Raw Score": row['best_score'],
                "Confidence (/100)": conf
            })

with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Summary written to {output_csv}")
