import os
import re
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from lxml import etree
from tqdm import tqdm

from ultralytics import YOLO
from paddleocr import PaddleOCR


PHOTOS_DIR = "photos"
ANNOTATIONS_XML = "annotations.xml"

VAL_SPLIT = 0.30
RANDOM_SEED = 42

OUTPUT_DIR = "outputs"
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

NUM_WORKERS = max(2, mp.cpu_count() - 1)
SPEED_SAMPLE_SIZE = 100


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)


def normalize_plate_text(text: str) -> str:
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


@dataclass
class PlateAnnotation:
    filename: str
    width: int
    height: int
    gt_bbox: Tuple[float, float, float, float]
    gt_text: str


@dataclass
class PlateResult:
    filename: str
    pred_text: str
    pred_bbox: Optional[Tuple[int, int, int, int]]
    iou: float
    is_correct: bool
    elapsed_sec: float


def parse_annotations(xml_path: str) -> List[PlateAnnotation]:
    tree = etree.parse(xml_path)
    root = tree.getroot()

    items: List[PlateAnnotation] = []
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
        gt_text = attr.text.strip() if attr is not None and attr.text else ""

        items.append(
            PlateAnnotation(
                filename=filename,
                width=width,
                height=height,
                gt_bbox=(xtl, ytl, xbr, ybr),
                gt_text=normalize_plate_text(gt_text),
            )
        )

    return items


def split_train_val(items: List[PlateAnnotation]) -> Tuple[List[PlateAnnotation], List[PlateAnnotation]]:
    import random

    random.seed(RANDOM_SEED)
    shuffled = items[:]
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_val = max(1, int(round(VAL_SPLIT * n_total)))

    val_items = shuffled[:n_val]
    train_items = shuffled[n_val:]
    return train_items, val_items


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, int(round(bx1)))
    iy1 = max(ay1, int(round(by1)))
    ix2 = min(ax2, int(round(bx2)))
    iy2 = min(ay2, int(round(by2)))

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, int(round(bx2 - bx1))) * max(0, int(round(by2 - by1)))
    union = a_area + b_area - inter

    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def clip_bbox_xyxy(b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return (x1, y1, x2, y2)


def yolo_best_bbox_from_result(result) -> Optional[Tuple[int, int, int, int]]:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None

    if conf is None:
        best = boxes_xyxy[0]
    else:
        best = boxes_xyxy[int(np.argmax(conf))]

    x1, y1, x2, y2 = best.tolist()
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def preprocess_for_paddleocr(plate_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rgb = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)
    return rgb


def paddleocr_read_plate(ocr: PaddleOCR, plate_bgr: np.ndarray) -> str:
    img = preprocess_for_paddleocr(plate_bgr)

    result = ocr.ocr(img, det=False, rec=True, cls=False)
    if not result:
        return ""

    best_text = ""
    best_conf = -1.0

    for item in result:
        if not item or len(item) < 2:
            continue
        text = item[0]
        conf = float(item[1])

        norm = normalize_plate_text(text)
        if len(norm) == 0:
            continue

        score = conf * min(1.0, len(norm) / 7.0)
        if score > best_conf:
            best_conf = score
            best_text = norm

    return best_text


def draw_visualization(
    img_bgr: np.ndarray,
    gt_bbox: Tuple[float, float, float, float],
    pred_bbox: Optional[Tuple[int, int, int, int]],
    gt_text: str,
    pred_text: str,
    iou_val: float,
) -> np.ndarray:
    vis = img_bgr.copy()

    gx1, gy1, gx2, gy2 = [int(round(v)) for v in gt_bbox]
    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)

    if pred_bbox is not None:
        px1, py1, px2, py2 = pred_bbox
        cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 0), 2)

    line1 = f"GT: {gt_text}"
    line2 = f"PRED: {pred_text} | IoU: {iou_val:.2f}"

    x0, y0 = 20, 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2

    (tw1, th1), _ = cv2.getTextSize(line1, font, scale, thickness)
    (tw2, th2), _ = cv2.getTextSize(line2, font, scale, thickness)
    box_w = max(tw1, tw2) + 20
    box_h = th1 + th2 + 30

    cv2.rectangle(vis, (x0 - 10, y0 - 30), (x0 - 10 + box_w, y0 - 30 + box_h), (0, 0, 0), -1)
    cv2.putText(vis, line1, (x0, y0), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(vis, line2, (x0, y0 + th1 + 15), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return vis


def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0

    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50
    score = 0.7 * accuracy_norm + 0.3 * time_norm

    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2


def resolve_weights_path() -> str:
    env_path = os.environ.get("PLATE_WEIGHTS", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path

    candidates = [
        os.path.join("runs", "plate_detector", "weights", "best.pt"),
        os.path.join("runs", "detect", "plate_detector", "weights", "best.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        "Missing best.pt. Set env PLATE_WEIGHTS to the full path of best.pt "
        "or place it in runs/plate_detector/weights/best.pt"
    )


WORKER_MODEL = None
WORKER_OCR = None


def worker_init(use_gpu_ocr: bool) -> None:
    global WORKER_OCR

    import os
    os.environ["FLAGS_minloglevel"] = "3"

    from paddleocr import PaddleOCR

    WORKER_OCR = PaddleOCR(
        use_angle_cls=False,
        lang="en",
        use_gpu=use_gpu_ocr,
    )



def worker_process(task):
    filename, gt_bbox, gt_text, save_vis = task

    t0 = time.perf_counter()

    img_path = os.path.join(PHOTOS_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        elapsed = time.perf_counter() - t0
        return PlateResult(filename, "", None, 0.0, False, elapsed)

    h, w = img.shape[:2]

    pred = WORKER_MODEL.predict(source=img, verbose=False, conf=0.25, iou=0.45)
    pred_bbox = yolo_best_bbox_from_result(pred[0] if pred else None)
    if pred_bbox is not None:
        pred_bbox = clip_bbox_xyxy(pred_bbox, w, h)

    pred_text = ""
    if pred_bbox is not None:
        x1, y1, x2, y2 = pred_bbox
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            pred_text = paddleocr_read_plate(WORKER_OCR, roi)

    iou_val = iou_xyxy(pred_bbox, gt_bbox) if pred_bbox is not None else 0.0
    is_correct = (normalize_plate_text(pred_text) == gt_text)

    elapsed = time.perf_counter() - t0

    if save_vis:
        vis = draw_visualization(img, gt_bbox, pred_bbox, gt_text, pred_text, iou_val)
        out_path = os.path.join(VIS_DIR, f"vis_{filename}")
        cv2.imwrite(out_path, vis)

    return PlateResult(filename, pred_text, pred_bbox, iou_val, is_correct, elapsed)


def main() -> None:
    ensure_dirs()

    if not os.path.isfile(ANNOTATIONS_XML):
        raise FileNotFoundError("Missing annotations.xml in project root")
    if not os.path.isdir(PHOTOS_DIR):
        raise FileNotFoundError("Missing photos/ folder")

    weights_path = resolve_weights_path()
    print(f"Weights: {weights_path}")

    anns = parse_annotations(ANNOTATIONS_XML)
    existing: List[PlateAnnotation] = []
    for a in anns:
        p = os.path.join(PHOTOS_DIR, a.filename)
        if os.path.isfile(p):
            existing.append(a)

    if len(existing) == 0:
        raise RuntimeError("No images found in photos/ that match annotations.xml")

    _, val_items = split_train_val(existing)
    print(f"Val(test): {len(val_items)} (split {VAL_SPLIT})")

    vis_set = set([a.filename for a in val_items[:10]])

    eval_tasks = []
    for ann in val_items:
        eval_tasks.append((ann.filename, ann.gt_bbox, ann.gt_text, ann.filename in vis_set))

    use_gpu_ocr = False

    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=worker_init,
        initargs=(weights_path, use_gpu_ocr),
    ) as pool:
        results = list(tqdm(pool.imap(worker_process, eval_tasks), total=len(eval_tasks), desc="Eval on val"))

    correct = sum(1 for r in results if r.is_correct)
    accuracy = 100.0 * correct / max(1, len(results))
    mean_iou = float(np.mean([r.iou for r in results])) if results else 0.0

    speed_items = existing[: min(SPEED_SAMPLE_SIZE, len(existing))]
    speed_tasks = [(a.filename, a.gt_bbox, a.gt_text, False) for a in speed_items]

    t0 = time.perf_counter()
    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=worker_init,
        initargs=(weights_path, use_gpu_ocr),
    ) as pool:
        _ = list(pool.imap(worker_process, speed_tasks))
    speed_time = time.perf_counter() - t0

    grade = calculate_final_grade(accuracy, speed_time)

    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Val images: {len(val_items)}\n")
        f.write(f"OCR accuracy (%): {accuracy:.2f}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Time for {len(speed_items)} images (s): {speed_time:.2f}\n")
        f.write(f"Final grade: {grade:.1f}\n")
        f.write("\nPer-image details:\n")
        for r in sorted(results, key=lambda x: x.filename):
            f.write(f"{r.filename} | GT? | PRED={r.pred_text} | IoU={r.iou:.4f} | OK={int(r.is_correct)}\n")

    print(f"OCR accuracy: {accuracy:.2f}%")
    print(f"Mean IoU: {mean_iou:.3f}")
    print(f"Time for {len(speed_items)} images: {speed_time:.2f}s")
    print(f"Final grade: {grade:.1f}")
    print(f"Report saved: {report_path}")
    print(f"Visualizations: {VIS_DIR}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
