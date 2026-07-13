import os
import coremltools as cmt
from ultralytics import YOLO

# 1. Path to your freshly trained YOLO11n weights
#weights_path = "runs.yolo26m/detect/HockeyAI/yolo26m_hockey_1280/weights/best.pt"
weights_path = "runs/detect/HockeyAI/yolo26m_hockey_1280/weights/last.pt"

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Could not find trained weights at: {weights_path}")

print("⚡ Loading custom YOLO11n hockey model...")
model = YOLO(weights_path)

# 2. Compile to CoreML with strict hockey optimization constraints
print("🚀 Compiling to CoreML (1280px, INT8 Weight-Only, Stable NMS)...")
export_path = model.export(
    format="coreml",
    imgsz=(720, 1280), # Strict 1280 constraint for multi-pixel puck visibility
    half=True,          # Ensures the operational layer precision stays natively in FP16
    end2end=False,
    nms=True
)

print(f"🎉 Success! Exported asset generated at: {export_path}")
