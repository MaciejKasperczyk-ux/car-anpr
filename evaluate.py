import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from lxml import etree
from tqdm import tqdm
from ultralytics import YOLO
import pytesseract

from plate_rules import clean_plate, match_plate


PHOTOS_DIR = "photos"
ANNOTATIONS_XML = "annotations.xml"

DATASET_YOLO_DIR = "dataset_yolo"
YOLO_VAL_IMAGES_DIR = os.path.join(DATASET_YOLO_DIR, "images", "val")

OUTPUT_DIR = "outputs"
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

SPEED_SAMPLE_SIZE = 100

YOLO_CONF = 0.25
YOLO_IOU = 0.45

OCR_MIN_LEN_TRIGGER_FALLBACK = 5

USE_OCR_CACHE_FOR_EVAL = True
USE_OCR_CACHE_FOR_SPEED = False
OCR_CACHE_MAX_ITEMS = 5000

IOU_HIT_THRESHOLD = 0.50

VIS_COUNT = 0
VIS_ONLY_ERRORS = True

OCR_WORKERS = max(1, (os.cpu_count() or 4) // 2)


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)


@dataclass
class PlateAnnotation:
    filename: str
    width: int
    height: int
    gt_bbox: Tuple[float, float, float, float]
    gt_text: str
    rotation: float


@dataclass
class PlateResult:
    filename: str
    gt_text: str
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
        rot = float(box.get("rotation") or "0")

        attr = box.find(".//attribute[@name='plate number']")
        gt_text_raw = attr.text.strip() if attr is not None and attr.text else ""

        items.append(
            PlateAnnotation(
                filename=filename,
                width=width,
                height=height,
                gt_bbox=(xtl, ytl, xbr, ybr),
                gt_text=clean_plate(gt_text_raw),
                rotation=rot,
            )
        )

    return items


def load_val_filenames_from_yolo(images_val_dir: str) -> Set[str]:
    if not os.path.isdir(images_val_dir):
        raise FileNotFoundError(f"Missing YOLO val images folder: {images_val_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    names: Set[str] = set()

    for fn in os.listdir(images_val_dir):
        _, ext = os.path.splitext(fn)
        if ext.lower() in exts:
            names.add(fn)

    if not names:
        raise RuntimeError(f"No images found in: {images_val_dir}")

    return names


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


def preprocess_for_ocr(plate_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    target_w = 320
    scale = max(1.0, float(target_w) / max(1, w))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if float(np.mean(thr == 255)) < 0.35:
        thr = cv2.bitwise_not(thr)

    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    return thr


def tesseract_try(img_bin: np.ndarray, psm: int) -> str:
    cfg = (
        f"--oem 1 --psm {psm} "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        "-c load_system_dawg=0 -c load_freq_dawg=0 "
        "-c preserve_interword_spaces=1 "
    )
    raw = pytesseract.image_to_string(img_bin, lang="eng", config=cfg)
    return clean_plate(raw)


def tesseract_read_plate(plate_bgr: np.ndarray, fast_only: bool) -> str:
    img = preprocess_for_ocr(plate_bgr)

    cand7 = tesseract_try(img, 7)
    if fast_only:
        return cand7

    if len(cand7) >= OCR_MIN_LEN_TRIGGER_FALLBACK:
        return cand7

    cand8 = tesseract_try(img, 8)
    cand6 = tesseract_try(img, 6)

    candidates = [cand7, cand8, cand6]
    candidates = [c for c in candidates if c]

    if not candidates:
        return ""

    candidates.sort(key=lambda x: (len(x), x), reverse=True)
    return candidates[0]


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
    line2 = f"PRED: {pred_text} IoU: {iou_val:.2f}"

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
        "or place it in runs/plate_detector/weights/best.pt or runs/detect/plate_detector/weights/best.pt"
    )


MODEL: Optional[YOLO] = None
OCR_CACHE: Dict[Tuple, str] = {}


def init_inference(weights_path: str) -> None:
    global MODEL
    MODEL = YOLO(weights_path)
    try:
        MODEL.fuse()
    except Exception:
        pass


def ocr_cached(filename: str, bbox: Tuple[int, int, int, int], roi: np.ndarray, use_cache: bool, fast_only: bool) -> str:
    if not use_cache:
        return tesseract_read_plate(roi, fast_only=fast_only)

    key = (filename, bbox[0], bbox[1], bbox[2], bbox[3], int(fast_only))
    if key in OCR_CACHE:
        return OCR_CACHE[key]

    text = tesseract_read_plate(roi, fast_only=fast_only)

    if len(OCR_CACHE) >= OCR_CACHE_MAX_ITEMS:
        OCR_CACHE.clear()

    OCR_CACHE[key] = text
    return text


def detect_bboxes_batch(filenames: List[str]) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
    assert MODEL is not None
    paths = [os.path.join(PHOTOS_DIR, fn) for fn in filenames]

    preds = MODEL.predict(
        source=paths,
        verbose=False,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        device="cpu",
        batch=16,
    )

    out: Dict[str, Optional[Tuple[int, int, int, int]]] = {}
    for fn, pred in zip(filenames, preds):
        out[fn] = yolo_best_bbox_from_result(pred)
    return out


def run_single_with_bbox(
    filename: str,
    gt_bbox: Tuple[float, float, float, float],
    gt_text: str,
    pred_bbox: Optional[Tuple[int, int, int, int]],
    save_vis: bool,
    use_cache: bool,
    fast_only: bool,
) -> PlateResult:
    t0 = time.perf_counter()

    img_path = os.path.join(PHOTOS_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        elapsed = time.perf_counter() - t0
        return PlateResult(filename, gt_text, "", None, 0.0, False, elapsed)

    h, w = img.shape[:2]

    if pred_bbox is not None:
        pred_bbox = clip_bbox_xyxy(pred_bbox, w, h)

    pred_text_raw = ""
    if pred_bbox is not None:
        x1, y1, x2, y2 = pred_bbox
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            pred_text_raw = ocr_cached(filename, pred_bbox, roi, use_cache=use_cache, fast_only=fast_only)

    iou_val = iou_xyxy(pred_bbox, gt_bbox) if pred_bbox is not None else 0.0

    ok, best_cand = match_plate(gt_text, pred_text_raw)
    pred_final = clean_plate(best_cand if best_cand else pred_text_raw)

    elapsed = time.perf_counter() - t0

    if save_vis:
        vis = draw_visualization(img, gt_bbox, pred_bbox, gt_text, pred_final, iou_val)
        out_path = os.path.join(VIS_DIR, f"vis_{filename}")
        cv2.imwrite(out_path, vis)

    return PlateResult(filename, gt_text, pred_final, pred_bbox, iou_val, ok, elapsed)


def _speed_task(ann: PlateAnnotation, pred_bbox: Optional[Tuple[int, int, int, int]]) -> None:
    _ = run_single_with_bbox(
        filename=ann.filename,
        gt_bbox=ann.gt_bbox,
        gt_text=ann.gt_text,
        pred_bbox=pred_bbox,
        save_vis=False,
        use_cache=USE_OCR_CACHE_FOR_SPEED,
        fast_only=True,
    )


def main() -> None:
    ensure_dirs()

    if not os.path.isfile(ANNOTATIONS_XML):
        raise FileNotFoundError("Missing annotations.xml in project root")
    if not os.path.isdir(PHOTOS_DIR):
        raise FileNotFoundError("Missing photos folder")
    if not os.path.isdir(YOLO_VAL_IMAGES_DIR):
        raise FileNotFoundError(f"Missing YOLO val images folder: {YOLO_VAL_IMAGES_DIR}")

    val_filenames = load_val_filenames_from_yolo(YOLO_VAL_IMAGES_DIR)

    weights_path = resolve_weights_path()
    print(f"Weights: {weights_path}")

    init_inference(weights_path)

    anns = parse_annotations(ANNOTATIONS_XML)
    ann_by_name: Dict[str, PlateAnnotation] = {a.filename: a for a in anns}

    val_items: List[PlateAnnotation] = []
    for fn in sorted(val_filenames):
        if fn in ann_by_name and os.path.isfile(os.path.join(PHOTOS_DIR, fn)):
            val_items.append(ann_by_name[fn])

    if not val_items:
        raise RuntimeError("No validation images matched between YOLO val set, photos, and annotations.xml")

    print(f"Val(test): {len(val_items)} (from {YOLO_VAL_IMAGES_DIR})")

    val_names = [a.filename for a in val_items]
    det_map_val = detect_bboxes_batch(val_names)

    results: List[PlateResult] = []
    vis_left = VIS_COUNT

    for ann in tqdm(val_items, total=len(val_items), desc="Eval on val"):
        pred_bbox = det_map_val.get(ann.filename)

        r = run_single_with_bbox(
            filename=ann.filename,
            gt_bbox=ann.gt_bbox,
            gt_text=ann.gt_text,
            pred_bbox=pred_bbox,
            save_vis=False,
            use_cache=USE_OCR_CACHE_FOR_EVAL,
            fast_only=False,
        )

        do_vis = False
        if vis_left > 0:
            if (not VIS_ONLY_ERRORS) or (VIS_ONLY_ERRORS and not r.is_correct):
                do_vis = True

        if do_vis:
            _ = run_single_with_bbox(
                filename=ann.filename,
                gt_bbox=ann.gt_bbox,
                gt_text=ann.gt_text,
                pred_bbox=pred_bbox,
                save_vis=True,
                use_cache=USE_OCR_CACHE_FOR_EVAL,
                fast_only=False,
            )
            vis_left -= 1

        results.append(r)

    correct = sum(1 for r in results if r.is_correct)
    accuracy = 100.0 * correct / max(1, len(results))
    mean_iou = float(np.mean([r.iou for r in results])) if results else 0.0

    iou_hits = sum(1 for r in results if r.iou >= IOU_HIT_THRESHOLD)
    iou_hit_rate = 100.0 * iou_hits / max(1, len(results))

    speed_pool: List[PlateAnnotation] = []
    for a in anns:
        p = os.path.join(PHOTOS_DIR, a.filename)
        if os.path.isfile(p):
            speed_pool.append(a)

    if not speed_pool:
        raise RuntimeError("No images found in photos matching annotations.xml for speed test")

    speed_items = speed_pool[: min(SPEED_SAMPLE_SIZE, len(speed_pool))]
    speed_names = [a.filename for a in speed_items]

    det_map_speed = detect_bboxes_batch(speed_names)

    OCR_CACHE.clear()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as ex:
        futures = []
        for ann in speed_items:
            futures.append(ex.submit(_speed_task, ann, det_map_speed.get(ann.filename)))

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Speed test"):
            pass
    speed_time = time.perf_counter() - t0

    grade = calculate_final_grade(accuracy, speed_time)

    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Val images: {len(val_items)}\n")
        f.write(f"OCR accuracy (%): {accuracy:.2f}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"IoU >= {IOU_HIT_THRESHOLD:.2f} (%): {iou_hit_rate:.2f}\n")
        f.write(f"Time for {len(speed_items)} images (s): {speed_time:.2f}\n")
        f.write(f"Final grade: {grade:.1f}\n")
        f.write("\nPer-image details:\n")
        for r in sorted(results, key=lambda x: x.filename):
            f.write(f"{r.filename} | GT={r.gt_text} | PRED={r.pred_text} | IoU={r.iou:.4f} | OK={int(r.is_correct)}\n")

    print(f"OCR accuracy: {accuracy:.2f}%")
    print(f"Mean IoU: {mean_iou:.3f}")
    print(f"IoU >= {IOU_HIT_THRESHOLD:.2f}: {iou_hit_rate:.2f}%")
    print(f"Time for {len(speed_items)} images: {speed_time:.2f}s")
    print(f"Final grade: {grade:.1f}")
    print(f"Report saved: {report_path}")
    print(f"Visualizations: {VIS_DIR}")


if __name__ == "__main__":
    main()
