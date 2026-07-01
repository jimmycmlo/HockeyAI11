from ultralytics import YOLO

# Initialize your target lightweight Nano model
model = YOLO("yolo11n.pt")

results = model.train(
    data="hockey_ai.yaml",
    epochs=50,
    imgsz=1280,           
    device="mps",         
    
    # --- APPLE SILICON OPTIMIZED HARDWARE SETTINGS ---
    batch=4,              # Lower batch size keeps tensor allocations tightly inside M4 GPU cache
    amp=False,            # CRITICAL: Disabling AMP prevents performance-killing MPS data casting loops
    workers=4,            # 4 workers is the sweet spot for Apple Silicon unified data loading
    cache="ram",          # Keeps data locked directly into your fast RAM system
    
    # --- COMPACT PUCK CONFIGURATION PARAMS ---
    mosaic=0.3,           # Prevents slicing the puck across border blocks too heavily
    scale=0.8,            # Forces the network to practice detecting tiny features
    box=8.5,              # Tightens bounding box target coordinate precision
    cls=1.5,              # Increases category classification confidence
    close_mosaic=10,
    patience=15,
    project="HockeyAI",
    name="yolo11n_hockey_1280"
)
