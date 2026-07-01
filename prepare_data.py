import os
import shutil
import random
from pathlib import Path

# Setup paths
base_dir = Path("SHL")
frames_dir = base_dir / "frames"
annotations_dir = base_dir / "annotations"

output_dir = Path("processed_hockey")
for split in ["train", "val"]:
    (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Gather all matching image pairs
all_frames = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
valid_pairs = []

for frame in all_frames:
    base_name = os.path.splitext(frame)[0]
    annotation_file = f"{base_name}.txt"
    if (annotations_dir / annotation_file).exists():
        valid_pairs.append((frame, annotation_file))

# Shuffle and split (80% Train, 20% Val)
random.seed(42)
random.shuffle(valid_pairs)
split_idx = int(len(valid_pairs) * 0.8)
train_pairs = valid_pairs[:split_idx]
val_pairs = valid_pairs[split_idx:]

def move_files(pairs, split_name):
    for img, txt in pairs:
        shutil.copy(frames_dir / img, output_dir / "images" / split_name / img)
        shutil.copy(annotations_dir / txt, output_dir / "labels" / split_name / txt)

move_files(train_pairs, "train")
move_files(val_pairs, "val")

print(f"Dataset Split Complete! Train: {len(train_pairs)} frames | Val: {len(val_pairs)} frames")
