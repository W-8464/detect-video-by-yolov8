import argparse
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Obj:
    cls: int
    pts_norm: np.ndarray  # (4,2) in [0,1] relative to full image


def parse_data_yaml_names(data_yaml_path: Path) -> Optional[List[str]]:
    try:
        txt = data_yaml_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = re.search(r"names\s*:\s*\[(.*?)\]", txt, flags=re.DOTALL)
    if not m:
        return None
    inner = m.group(1)
    # Extract quoted strings (single or double quotes)
    names = re.findall(r"'([^']*)'|\"([^\"]*)\"", inner)
    out: List[str] = []
    for a, b in names:
        s = a if a else b
        if s.strip():
            out.append(s.strip())
    return out or None


def read_obb_label_file(label_path: Path) -> List[Obj]:
    if not label_path.exists():
        return []
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    out: List[Obj] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 9:
            continue
        cls = int(float(parts[0]))
        coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
        out.append(Obj(cls=cls, pts_norm=coords))
    return out


def pts_norm_to_abs(pts_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = pts_norm.copy()
    pts[:, 0] *= float(w)
    pts[:, 1] *= float(h)
    return pts


def aabb_from_pts(pts_xy: np.ndarray) -> Tuple[float, float, float, float]:
    xs = pts_xy[:, 0]
    ys = pts_xy[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def center_of_pts(pts_xy: np.ndarray) -> Tuple[float, float]:
    return float(pts_xy[:, 0].mean()), float(pts_xy[:, 1].mean())


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
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def expand_clip_aabb(a: Tuple[float, float, float, float], w: int, h: int, pad: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = a
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * pad
    bh = (y2 - y1) * pad
    nx1 = int(round(cx - bw / 2.0))
    ny1 = int(round(cy - bh / 2.0))
    nx2 = int(round(cx + bw / 2.0))
    ny2 = int(round(cy + bh / 2.0))
    nx1 = max(0, min(w - 1, nx1))
    ny1 = max(0, min(h - 1, ny1))
    nx2 = max(1, min(w, nx2))
    ny2 = max(1, min(h, ny2))
    if nx2 <= nx1 + 1:
        nx2 = min(w, nx1 + 2)
    if ny2 <= ny1 + 1:
        ny2 = min(h, ny1 + 2)
    return nx1, ny1, nx2, ny2


def write_yolo_obb_label(objs: List[Tuple[int, np.ndarray]], out_path: Path) -> None:
    # objs: list of (cls, pts_norm (4,2) in crop space)
    lines: List[str] = []
    for cls, pts in objs:
        pts = np.clip(pts, 0.0, 1.0)
        flat = pts.reshape(-1).tolist()
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in flat))
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create ROI OBB dataset cropped around hand/tweezers.")
    ap.add_argument("--src", default="datasets/yolo_format", help="Source dataset root (images/ labels/).")
    ap.add_argument("--dst", default="datasets/roi_format", help="Destination dataset root.")
    ap.add_argument("--splits", default="train,val", help="Comma-separated splits to process.")
    ap.add_argument("--hand-cls", type=int, default=0, help="Class id for hand.")
    ap.add_argument("--tweezers-cls", type=int, default=5, help="Class id for tweezers.")
    ap.add_argument("--shielding-cls", type=int, default=6, help="Class id for shielding.")
    ap.add_argument("--gasket-cls", type=int, default=8, help="Class id for gasket.")
    ap.add_argument("--keep-classes", default="0,5,6,8", help="Comma-separated class ids to keep in ROI labels.")
    ap.add_argument("--pad", type=float, default=1.6, help="Padding multiplier for ROI box size.")
    ap.add_argument("--inhand-iou", type=float, default=0.05, help="AABB IoU threshold to count as in-hand.")
    ap.add_argument("--neg-keep", type=float, default=0.2, help="Keep probability for negative ROIs.")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth image to reduce near-duplicates.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling negatives.")
    ap.add_argument("--max-images", type=int, default=0, help="Optional cap per split (0=all).")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    keep_classes = {int(x.strip()) for x in args.keep_classes.split(",") if x.strip()}
    hand_cls = args.hand_cls
    tweezers_cls = args.tweezers_cls
    target_small = {args.shielding_cls, args.gasket_cls}

    # Prepare output dirs
    for sp in splits:
        (dst / "images" / sp).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / sp).mkdir(parents=True, exist_ok=True)

    # Optional data_roi.yaml (subset names if we can parse)
    names_full = parse_data_yaml_names(Path("data.yaml"))
    if names_full:
        # keep order by class id for kept classes
        kept_sorted = sorted(keep_classes)
        # Re-index classes to 0..k-1 for ROI training (recommended)
        new_names = []
        for cid in kept_sorted:
            if 0 <= cid < len(names_full):
                new_names.append(names_full[cid])
            else:
                new_names.append(f"class_{cid}")
        data_roi = dst / "data_roi.yaml"
        # Ultralytics resolves relative paths from the YAML file directory.
        # Write paths relative to dst/ to avoid duplicated prefixes when YAML lives inside dst.
        data_roi.write_text(
            "\n".join(
                [
                    "train: images/train",
                    "val: images/val",
                    "",
                    f"nc: {len(kept_sorted)}",
                    "names: [" + ", ".join(repr(n) for n in new_names) + "]",
                    "",
                    "# NOTE: ROI labels are re-indexed to match this names list.",
                    "# Original class ids were: " + ", ".join(str(c) for c in kept_sorted),
                    "",
                ]
            ),
            encoding="utf-8",
        )

    # Mapping original class id -> new contiguous id
    kept_sorted = sorted(keep_classes)
    cls_remap = {cid: i for i, cid in enumerate(kept_sorted)}

    summary = {sp: {"images": 0, "rois": 0, "pos": 0, "neg": 0, "skipped_no_hand": 0} for sp in splits}

    for sp in splits:
        img_dir = src / "images" / sp
        lbl_dir = src / "labels" / sp
        if not img_dir.exists():
            continue

        # Use sorted list for determinism
        img_paths = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if args.stride > 1:
            img_paths = img_paths[:: args.stride]
        if args.max_images and args.max_images > 0:
            img_paths = img_paths[: args.max_images]

        out_img_dir = dst / "images" / sp
        out_lbl_dir = dst / "labels" / sp

        for img_path in img_paths:
            summary[sp]["images"] += 1
            label_path = lbl_dir / (img_path.stem + ".txt")
            objs = read_obb_label_file(label_path)
            if not objs:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Separate by type
            hands = []
            smalls = []
            for o in objs:
                pts_abs = pts_norm_to_abs(o.pts_norm, w=w, h=h)
                aabb = aabb_from_pts(pts_abs)
                if o.cls in (hand_cls, tweezers_cls):
                    hands.append((o.cls, pts_abs, aabb))
                if o.cls in target_small:
                    smalls.append((o.cls, pts_abs, aabb))

            if not hands:
                summary[sp]["skipped_no_hand"] += 1
                continue

            # Create one ROI per hand/tweezers instance
            for k, (hcls, hpts_abs, haabb) in enumerate(hands):
                rx1, ry1, rx2, ry2 = expand_clip_aabb(haabb, w=w, h=h, pad=args.pad)
                crop = img[ry1:ry2, rx1:rx2]
                ch, cw = crop.shape[:2]
                if ch < 2 or cw < 2:
                    continue

                # Determine positive ROI: shielding/gasket overlaps hand/tweezers AABB in this ROI
                is_pos = False
                for scls, spts_abs, saabb in smalls:
                    if iou_aabb(haabb, saabb) >= args.inhand_iou:
                        is_pos = True
                        break
                    cx, cy = center_of_pts(spts_abs)
                    hx1, hy1, hx2, hy2 = haabb
                    if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
                        is_pos = True
                        break

                if not is_pos:
                    if random.random() > float(args.neg_keep):
                        continue

                # Collect objs whose center is inside ROI, transform pts to crop-relative normalized
                kept_for_roi: List[Tuple[int, np.ndarray]] = []
                for o in objs:
                    if o.cls not in keep_classes:
                        continue
                    pts_abs = pts_norm_to_abs(o.pts_norm, w=w, h=h)
                    cx, cy = center_of_pts(pts_abs)
                    if not (rx1 <= cx < rx2 and ry1 <= cy < ry2):
                        continue
                    pts_crop = pts_abs.copy()
                    pts_crop[:, 0] = (pts_crop[:, 0] - float(rx1)) / float(cw)
                    pts_crop[:, 1] = (pts_crop[:, 1] - float(ry1)) / float(ch)
                    # Remap class id to contiguous for ROI training
                    kept_for_roi.append((cls_remap[o.cls], pts_crop.astype(np.float32)))

                # Always keep the ROI hand/tweezers itself if present (so model can learn context)
                if not kept_for_roi:
                    continue

                out_stem = f"{img_path.stem}_roi_{hcls}_{k:02d}"
                out_img_path = out_img_dir / f"{out_stem}.png"
                out_lbl_path = out_lbl_dir / f"{out_stem}.txt"

                cv2.imwrite(str(out_img_path), crop)
                write_yolo_obb_label(kept_for_roi, out_lbl_path)

                summary[sp]["rois"] += 1
                if is_pos:
                    summary[sp]["pos"] += 1
                else:
                    summary[sp]["neg"] += 1

    print("✅ ROI dataset created:", str(dst))
    for sp in splits:
        s = summary[sp]
        if s["images"] == 0:
            continue
        print(
            f"- {sp}: images_seen={s['images']} rois_written={s['rois']} pos={s['pos']} neg={s['neg']} skipped_no_hand={s['skipped_no_hand']}"
        )
    if (dst / "data_roi.yaml").exists():
        print("🧾 Wrote:", str(dst / "data_roi.yaml"))


if __name__ == "__main__":
    main()

