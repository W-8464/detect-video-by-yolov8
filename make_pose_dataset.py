import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


def parse_pose_label(label_path: Path) -> List[dict]:
    """
    Parse YOLO-pose label file (CVAT export).
    Format per line: <cls> <cx> <cy> <w> <h> <kx1> <ky1> <kv1> ... <kxN> <kyN> <kvN>
    Returns list of dicts with keys: cls, bbox (cx,cy,w,h), kpts (N,3)
    """
    if not label_path.exists():
        return []
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    out = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 6:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        kpt_flat = [float(x) for x in parts[5:]]
        if len(kpt_flat) % 3 != 0:
            continue
        n_kpts = len(kpt_flat) // 3
        kpts = np.array(kpt_flat, dtype=np.float32).reshape(n_kpts, 3)
        out.append({"cls": cls, "bbox": (cx, cy, w, h), "kpts": kpts})
    return out


def write_pose_label(instances: List[dict], out_path: Path) -> None:
    """Write YOLO-pose label file."""
    lines = []
    for inst in instances:
        cls = inst["cls"]
        cx, cy, w, h = inst["bbox"]
        kpts = inst["kpts"]
        kpt_flat = kpts.reshape(-1).tolist()
        vals = [cls, cx, cy, w, h] + kpt_flat
        lines.append(" ".join(f"{v:.6f}" for v in vals))
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def bbox_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[int, int, int, int]:
    x1 = int(round((cx - w / 2)))
    y1 = int(round((cy - h / 2)))
    x2 = int(round((cx + w / 2)))
    y2 = int(round((cy + h / 2)))
    return x1, y1, x2, y2


def expand_clip_bbox(cx: float, cy: float, w: float, h: float,
                     img_w: int, img_h: int, pad: float) -> Tuple[int, int, int, int]:
    bw = w * pad
    bh = h * pad
    nx1 = int(round(cx - bw / 2))
    ny1 = int(round(cy - bh / 2))
    nx2 = int(round(cx + bw / 2))
    ny2 = int(round(cy + bh / 2))
    nx1 = max(0, min(img_w - 2, nx1))
    ny1 = max(0, min(img_h - 2, ny1))
    nx2 = max(nx1 + 2, min(img_w, nx2))
    ny2 = max(ny1 + 2, min(img_h, ny2))
    return nx1, ny1, nx2, ny2


def main():
    ap = argparse.ArgumentParser(description="Prepare YOLO-pose dataset from CVAT exports.")
    ap.add_argument("--src", nargs="+", required=True,
                    help="Source dataset folders (CVAT YOLO-pose exports).")
    ap.add_argument("--dst", default="datasets/pose_format",
                    help="Destination dataset root.")
    ap.add_argument("--val-ratio", type=float, default=0.15,
                    help="Validation split ratio.")
    ap.add_argument("--crop-hand", action="store_true",
                    help="Crop hand regions from full-frame images (recommended).")
    ap.add_argument("--pad", type=float, default=1.6,
                    help="Padding multiplier for hand crop (only with --crop-hand).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for train/val split.")
    ap.add_argument("--kpt-shape", default="16,3",
                    help="Keypoint shape: N,dim (e.g. 16,3).")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dst = Path(args.dst)
    kpt_n, kpt_dim = [int(x.strip()) for x in args.kpt_shape.split(",")]

    # Collect all labeled frames
    all_samples: List[Tuple[Path, Path, str]] = []  # (image_path, label_path, source_name)

    for src_dir in args.src:
        src = Path(src_dir)
        if not src.exists():
            print(f"⚠️  Source not found, skipping: {src}")
            continue

        # Find images and labels (handle nested train/ subdir from CVAT export)
        img_dir = src / "images" / "train"
        lbl_dir = src / "labels" / "train"
        if not img_dir.exists():
            img_dir = src / "images"
            lbl_dir = src / "labels"

        if not img_dir.exists():
            print(f"⚠️  No images/ found in {src}, skipping")
            continue

        img_paths = sorted([p for p in img_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

        skipped_no_label = 0
        for img_path in img_paths:
            label_path = lbl_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                skipped_no_label += 1
                continue
            instances = parse_pose_label(label_path)
            if not instances:
                skipped_no_label += 1
                continue
            all_samples.append((img_path, label_path, src.name))

        print(f"📁 {src.name}: {len(img_paths)} images, "
              f"{len(img_paths) - skipped_no_label} labeled, "
              f"{skipped_no_label} skipped (no labels)")

    if not all_samples:
        print("❌ No labeled samples found. Exiting.")
        return

    print(f"\n📊 Total labeled frames: {len(all_samples)}")

    # Shuffle and split
    random.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * args.val_ratio))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    print(f"🔀 Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Write output
    for split, samples in [("train", train_samples), ("val", val_samples)]:
        out_img_dir = dst / "images" / split
        out_lbl_dir = dst / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path, src_name in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            instances = parse_pose_label(lbl_path)

            if args.crop_hand:
                # Crop each hand instance into separate samples
                for k, inst in enumerate(instances):
                    cx, cy, bw, bh = inst["bbox"]
                    # Convert normalized bbox to absolute pixels
                    cx_abs = cx * w
                    cy_abs = cy * h
                    bw_abs = bw * w
                    bh_abs = bh * h

                    rx1, ry1, rx2, ry2 = expand_clip_bbox(cx_abs, cy_abs, bw_abs, bh_abs, w, h, args.pad)
                    crop = img[ry1:ry2, rx1:rx2]
                    ch, cw = crop.shape[:2]
                    if ch < 4 or cw < 4:
                        continue

                    # Transform keypoints from full-frame normalized to crop normalized
                    kpts = inst["kpts"].copy()
                    kpts[:, 0] = (kpts[:, 0] * w - rx1) / cw
                    kpts[:, 1] = (kpts[:, 1] * h - ry1) / ch
                    kpts[:, 0] = np.clip(kpts[:, 0], 0.0, 1.0)
                    kpts[:, 1] = np.clip(kpts[:, 1], 0.0, 1.0)

                    # New bbox covering whole crop
                    new_inst = {
                        "cls": inst["cls"],
                        "bbox": (0.5, 0.5, 1.0, 1.0),
                        "kpts": kpts,
                    }

                    out_stem = f"{src_name}_{img_path.stem}_hand{k:02d}"
                    out_img_path = out_img_dir / f"{out_stem}.png"
                    out_lbl_path = out_lbl_dir / f"{out_stem}.txt"

                    cv2.imwrite(str(out_img_path), crop)
                    write_pose_label([new_inst], out_lbl_path)
            else:
                # Keep full-frame images as-is
                out_stem = f"{src_name}_{img_path.stem}"
                out_img_path = out_img_dir / f"{out_stem}{img_path.suffix}"
                out_lbl_path = out_lbl_dir / f"{out_stem}.txt"
                cv2.imwrite(str(out_img_path), img)
                write_pose_label(instances, out_lbl_path)

    # Count output
    total_hand_instances = 0
    for split in ["train", "val"]:
        img_dir = dst / "images" / split
        if img_dir.exists():
            n = len(list(img_dir.iterdir()))
            print(f"📦 {split}: {n} images")
            if args.crop_hand:
                total_hand_instances += n

    if args.crop_hand:
        print(f"🖐️  Total hand crops: {total_hand_instances}")

    # Write data.yaml
    data_yaml = dst / "data_pose.yaml"
    data_yaml.write_text(
        "\n".join([
            f"train: images/train",
            f"val: images/val",
            "",
            "nc: 1",
            "names: ['hand']",
            "",
            f"kpt_shape: [{kpt_n}, {kpt_dim}]",
            "",
            f"# Generated from: {', '.join(args.src)}",
            f"# Crop hand: {args.crop_hand}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"🧾 Wrote: {data_yaml}")
    print("✅ Done.")


if __name__ == "__main__":
    main()
