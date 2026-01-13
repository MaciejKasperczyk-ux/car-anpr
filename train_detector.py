import os
import torch
from ultralytics import YOLO

DATA_YAML = os.path.join("dataset_yolo", "data.yaml")

MODEL_NAME = "yolov8n.pt"
IMG_SIZE = 960
EPOCHS = 60
BATCH = 8
WORKERS = 2


def pick_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    if not os.path.isfile(DATA_YAML):
        raise FileNotFoundError("Missing dataset_yolo/data.yaml. Run prepare_dataset.py first.")

    device = pick_device()
    print(f"Using device: {device}")

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        workers=WORKERS,
        device=device,
        project="runs",
        name="plate_detector",
        exist_ok=True,
    )

    print("Training finished.")
    print("Check weights in runs/detect/plate_detector/weights/best.pt")


if __name__ == "__main__":
    main()
