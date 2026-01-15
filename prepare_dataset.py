import os
import shutil
import random
from typing import List, Tuple, Dict
from lxml import etree
import yaml

try:
    import kagglehub
except Exception:
    kagglehub = None

RANDOM_SEED = 42
VAL_SPLIT = 0.30
CLASS_NAME = "plate"
CLASS_ID = 0

DEFAULT_OUT_DIR = "dataset_yolo"

def ensure_empty_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def find_dataset_paths() -> Tuple[str, str]:
    photos_dir = os.path.join(os.getcwd(), "photos")
    annotations_xml = os.path.join(os.getcwd(), "annotations.xml")

    if os.path.isdir(photos_dir) and os.path.isfile(annotations_xml):
        return photos_dir, annotations_xml

    if kagglehub is None:
        raise FileNotFoundError(
            "Missing local dataset files: photos/ and annotations.xml. "
            "kagglehub is not installed, so automatic download is not available."
        )

    dataset_dir = kagglehub.dataset_download("piotrstefaskiue/poland-vehicle-license-plate-dataset")
    dataset_dir = os.path.abspath(dataset_dir)

    photos_dir = os.path.join(dataset_dir, "photos")
    annotations_xml = os.path.join(dataset_dir, "annotations.xml")

    if not os.path.isdir(photos_dir):
        raise FileNotFoundError(f"Missing photos directory in kagglehub cache: {photos_dir}")
    if not os.path.isfile(annotations_xml):
        raise FileNotFoundError(f"Missing annotations.xml in kagglehub cache: {annotations_xml}")

    return photos_dir, annotations_xml

def parse_annotations(xml_path: str) -> List[Dict]:
    tree = etree.parse(xml_path)
    root = tree.getroot()

    items: List[Dict] = []
    for image_node in root.findall(".//image"):
        filename = image_node.get("name")
        if not filename:
            continue

        width = int(image_node.get("width"))
        height = int(image_node.get("height"))

        box = image_node.find(".//box[@label='plate']")
        if box is None:
            continue

        xtl = float(box.get("xtl"))
        ytl = float(box.get("ytl"))
        xbr = float(box.get("xbr"))
        ybr = float(box.get("ybr"))
        rot = float(box.get("rotation") or "0")

        attr = box.find(".//attribute[@name='plate number']")
        plate_text = attr.text.strip() if attr is not None and attr.text else ""

        items.append(
            {
                "filename": filename,
                "width": width,
                "height": height,
                "bbox": (xtl, ytl, xbr, ybr),
                "rotation": rot,
                "plate_text": plate_text,
            }
        )

    return items

def bbox_to_yolo(bbox: Tuple[float, float, float, float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    xtl, ytl, xbr, ybr = bbox

    xtl = max(0.0, min(float(img_w - 1), xtl))
    ytl = max(0.0, min(float(img_h - 1), ytl))
    xbr = max(0.0, min(float(img_w - 1), xbr))
    ybr = max(0.0, min(float(img_h - 1), ybr))

    if xbr < xtl:
        xtl, xbr = xbr, xtl
    if ybr < ytl:
        ytl, ybr = ybr, ytl

    bw = max(1.0, (xbr - xtl))
    bh = max(1.0, (ybr - ytl))

    x_center = xtl + bw / 2.0
    y_center = ytl + bh / 2.0

    x_center /= float(img_w)
    y_center /= float(img_h)
    bw /= float(img_w)
    bh /= float(img_h)

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))

    return (x_center, y_center, bw, bh)

def write_yolo_label(label_path: str, class_id: int, yolo_box: Tuple[float, float, float, float]) -> None:
    xc, yc, w, h = yolo_box
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def copy_image(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def write_ground_truth_csv(items: List[Dict], out_dir: str) -> None:
    csv_path = os.path.join(out_dir, "gt_plates.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("filename,plate_text,rotation\n")
        for it in items:
            plate = (it.get("plate_text") or "").replace('"', '""')
            rot = float(it.get("rotation", 0.0))
            f.write(f"\"{it['filename']}\",\"{plate}\",{rot}\n")

def main() -> None:
    photos_dir, annotations_xml = find_dataset_paths()

    out_dir = DEFAULT_OUT_DIR
    items = parse_annotations(annotations_xml)

    existing: List[Dict] = []
    for it in items:
        src_path = os.path.join(photos_dir, it["filename"])
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

    print(f"Photos dir: {photos_dir}")
    print(f"Annotations: {annotations_xml}")
    print(f"Total: {n_total}")
    print(f"Train: {len(train_items)}")
    print(f"Val: {len(val_items)} (split {VAL_SPLIT})")

    ensure_empty_dir(out_dir)

    img_train_dir = os.path.join(out_dir, "images", "train")
    img_val_dir = os.path.join(out_dir, "images", "val")
    lbl_train_dir = os.path.join(out_dir, "labels", "train")
    lbl_val_dir = os.path.join(out_dir, "labels", "val")

    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(lbl_train_dir, exist_ok=True)
    os.makedirs(lbl_val_dir, exist_ok=True)

    def process_split(split_items: List[Dict], img_dir: str, lbl_dir: str) -> None:
        for it in split_items:
            src_img = os.path.join(photos_dir, it["filename"])
            dst_img = os.path.join(img_dir, it["filename"])

            base = os.path.splitext(it["filename"])[0]
            dst_lbl = os.path.join(lbl_dir, f"{base}.txt")

            yolo_box = bbox_to_yolo(it["bbox"], it["width"], it["height"])

            copy_image(src_img, dst_img)
            write_yolo_label(dst_lbl, CLASS_ID, yolo_box)

    process_split(train_items, img_train_dir, lbl_train_dir)
    process_split(val_items, img_val_dir, lbl_val_dir)

    data_yaml = {
        "path": os.path.abspath(out_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: CLASS_NAME},
    }

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    write_ground_truth_csv(existing, out_dir)

    print(f"Saved YOLO dataset to: {out_dir}")
    print(f"Saved data.yaml to: {yaml_path}")
    print(f"Saved GT CSV to: {os.path.join(out_dir, 'gt_plates.csv')}")

if __name__ == "__main__":
    main()
