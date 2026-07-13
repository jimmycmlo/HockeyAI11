from ultralytics import YOLO

# Load pretrained YOLO26m weights
model = YOLO("yolo26m.pt") #starting from scratch
#model = YOLO("runs/detect/HockeyAI/yolo26m_hockey_1280/weights/last.pt") # for 

results = model.train(
    data="hockey_ai.yaml",

    # Training duration
    resume=True,
    epochs=200,

    # Image size
    imgsz=1280,

    # Apple Silicon (M4 Pro) settings
    device="mps",
    batch=4,

    # Optimization
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3,

    # Project organization
    project="HockeyAI",
    name="yolo26m_hockey_1280",        
)