import os
import shutil
import random
from typing import List, Tuple, Dict
from lxml import etree
import yaml

PHOTOS_DIR = "photos"
ANNOTATIONS_XML = "annotations.xml"
OUT_DIR = "dataset_yolo"

RANDOM_SEED = 42
VAL_SPLIT = 0.30  # must be at least 0.30 for testing
CLASS_NAME = "plate"
CLASS_ID = 0


def ensure_empty_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def parse_annotations(xml_path: str) -> List[Dict]:
    tree = etree.parse(xml_path)
    root = tree.getroot()

    items: List[Dict] = []
    for image_node in root.findall(".//image"):
        filename = image_node.get("name")
        width = int(image_node.get("width"))
        height = int(image_node.get("height"))

        box = image_node.find(".//box[@label='plate']")
        if box is None:
            continue

        xtl = float(box.get("xtl"))
        ytl = float(box.get("ytl"))
        xbr = float(box.get("xbr"))
        ybr = float(box.get("ybr"))

        attr = box.find(".//attribute[@name='plate number']")
        plate_text = attr.text.strip() if attr is not None and attr.text else ""

        items.append(
            {
                "filename": filename,
                "width": width,
                "height": height,
                "bbox": (xtl, ytl, xbr, ybr),
                "plate_text": plate_text,
            }
        )

    return items


def bbox_to_yolo(
    bbox: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    xtl, ytl, xbr, ybr = bbox
    x_center = (xtl + xbr) / 2.0
    y_center = (ytl + ybr) / 2.0
    w = (xbr - xtl)
    h = (ybr - ytl)

    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return (x_center, y_center, w, h)


def write_yolo_label(label_path: str, class_id: int, yolo_box: Tuple[float, float, float, float]) -> None:
    xc, yc, w, h = yolo_box
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def copy_image(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    if not os.path.isfile(ANNOTATIONS_XML):
        raise FileNotFoundError(f"Missing {ANNOTATIONS_XML} in project root")

    if not os.path.isdir(PHOTOS_DIR):
        raise FileNotFoundError(f"Missing folder {PHOTOS_DIR}")

    items = parse_annotations(ANNOTATIONS_XML)

    existing: List[Dict] = []
    for it in items:
        src_path = os.path.join(PHOTOS_DIR, it["filename"])
        if os.path.isfile(src_path):
            existing.append(it)

    if len(existing) == 0:
        raise RuntimeError("No images matched between annotations.xml and photos/")

    random.seed(RANDOM_SEED)
    random.shuffle(existing)

    n_total = len(existing)
    n_val = max(1, int(round(VAL_SPLIT * n_total)))

    val_items = existing[:n_val]
    train_items = existing[n_val:]

    print(f"Total: {n_total}")
    print(f"Train: {len(train_items)}")
    print(f"Val: {len(val_items)} (split {VAL_SPLIT})")

    ensure_empty_dir(OUT_DIR)

    img_train_dir = os.path.join(OUT_DIR, "images", "train")
    img_val_dir = os.path.join(OUT_DIR, "images", "val")
    lbl_train_dir = os.path.join(OUT_DIR, "labels", "train")
    lbl_val_dir = os.path.join(OUT_DIR, "labels", "val")

    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(lbl_train_dir, exist_ok=True)
    os.makedirs(lbl_val_dir, exist_ok=True)

    def process_split(split_items: List[Dict], img_dir: str, lbl_dir: str) -> None:
        for it in split_items:
            src_img = os.path.join(PHOTOS_DIR, it["filename"])
            dst_img = os.path.join(img_dir, it["filename"])

            base = os.path.splitext(it["filename"])[0]
            dst_lbl = os.path.join(lbl_dir, f"{base}.txt")

            yolo_box = bbox_to_yolo(it["bbox"], it["width"], it["height"])

            copy_image(src_img, dst_img)
            write_yolo_label(dst_lbl, CLASS_ID, yolo_box)

    process_split(train_items, img_train_dir, lbl_train_dir)
    process_split(val_items, img_val_dir, lbl_val_dir)

    data_yaml = {
        "path": os.path.abspath(OUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {0: CLASS_NAME},
    }

    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"Saved YOLO dataset to: {OUT_DIR}")
    print(f"Saved data.yaml to: {yaml_path}")


if __name__ == "__main__":
    main()
