from ultralytics import YOLO

# 1. Point directly to the 'last.pt' checkpoint from your completed run
# Update the path to match your exact project/name folder structure
checkpoint_path = "HockeyAI/yolo11n_hockey_1280_fixed/weights/last.pt"

model = YOLO(checkpoint_path)

# 2. Resume training and extend the total epoch target
# Setting epochs=150 means it will train for 100 MORE epochs (50 completed + 100 new)
results = model.train(
    resume=True,
    epochs=150  # This is your NEW absolute target ceiling
)
