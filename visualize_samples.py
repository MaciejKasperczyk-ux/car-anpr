import os
import cv2
from lxml import etree
from ultralytics import YOLO

PHOTOS_DIR = "photos"
ANNOTATIONS_XML = "annotations.xml"
WEIGHTS_PATH = os.path.join("runs", "plate_detector", "weights", "best.pt")
OUT_PATH = os.path.join("outputs", "single_debug.jpg")


def find_annotation(filename: str):
    tree = etree.parse(ANNOTATIONS_XML)
    root = tree.getroot()
    for image_node in root.findall(".//image"):
        if image_node.get("name") != filename:
            continue
        box = image_node.find(".//box[@label='plate']")
        if box is None:
            return None
        xtl = float(box.get("xtl"))
        ytl = float(box.get("ytl"))
        xbr = float(box.get("xbr"))
        ybr = float(box.get("ybr"))
        attr = box.find(".//attribute[@name='plate number']")
        gt_text = attr.text.strip() if attr is not None and attr.text else ""
        return (xtl, ytl, xbr, ybr, gt_text)
    return None


def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    filename = "2.jpg"  # change to any image name from photos/
    ann = find_annotation(filename)
    if ann is None:
        raise RuntimeError("Annotation not found for selected image")

    xtl, ytl, xbr, ybr, gt_text = ann
    img_path = os.path.join(PHOTOS_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError("Image not found")

    model = YOLO(WEIGHTS_PATH)
    pred = model.predict(source=img, verbose=False, conf=0.25, iou=0.45)[0]

    if pred.boxes is not None and len(pred.boxes) > 0:
        boxes = pred.boxes.xyxy.cpu().numpy()
        conf = pred.boxes.conf.cpu().numpy()
        best = boxes[int(conf.argmax())]
        px1, py1, px2, py2 = [int(round(v)) for v in best.tolist()]
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)

    gx1, gy1, gx2, gy2 = [int(round(v)) for v in [xtl, ytl, xbr, ybr]]
    cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)

    cv2.putText(img, f"GT: {gt_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(OUT_PATH, img)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
