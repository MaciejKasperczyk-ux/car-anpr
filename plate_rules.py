import re
from typing import Tuple, Optional, List, Dict

ALNUM_RE = re.compile(r"[A-Z0-9]+")

CONFUSIONS: List[Tuple[str, str]] = [
    ("O", "0"),
    ("0", "O"),
    ("I", "1"),
    ("1", "I"),
    ("Z", "2"),
    ("2", "Z"),
    ("S", "5"),
    ("5", "S"),
    ("B", "8"),
    ("8", "B"),
    ("G", "6"),
    ("6", "G"),
]

def clean_plate(s: str) -> str:
    if not s:
        return ""
    s = s.upper()
    parts = ALNUM_RE.findall(s)
    if not parts:
        return ""
    return "".join(parts)

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def generate_trim_candidates(s: str, min_len: int = 4, max_trim: int = 2) -> List[str]:
    s = clean_plate(s)
    if not s:
        return []
    cands = {s}

    for k in range(1, max_trim + 1):
        if len(s) - k >= min_len:
            cands.add(s[k:])
        if len(s) - k >= min_len:
            cands.add(s[:-k])

    for k1 in range(1, max_trim + 1):
        for k2 in range(1, max_trim + 1):
            if len(s) - (k1 + k2) >= min_len:
                cands.add(s[k1:-k2])

    return sorted(cands, key=lambda x: (len(x), x), reverse=True)

def generate_confusion_candidates(s: str, max_changes: int = 2) -> List[str]:
    s = clean_plate(s)
    if not s:
        return []

    out = {s}
    chars = list(s)

    for i in range(len(chars)):
        for a, b in CONFUSIONS:
            if chars[i] == a:
                t = chars[:]
                t[i] = b
                out.add("".join(t))

    if max_changes >= 2:
        first_layer = list(out)
        for base in first_layer:
            base_chars = list(base)
            for i in range(len(base_chars)):
                for a, b in CONFUSIONS:
                    if base_chars[i] == a:
                        t = base_chars[:]
                        t[i] = b
                        out.add("".join(t))

    return list(out)

def pick_best_against_gt(gt: str, pred: str) -> Tuple[int, str]:
    gt = clean_plate(gt)
    pred = clean_plate(pred)

    if not gt and not pred:
        return 0, ""
    if not gt:
        return len(pred), pred
    if not pred:
        return len(gt), pred

    candidates = set()

    for c in generate_trim_candidates(pred, min_len=max(4, len(gt) - 2), max_trim=2):
        candidates.add(c)
        for cc in generate_confusion_candidates(c, max_changes=2):
            candidates.add(cc)

    candidates.add(pred)

    best = pred
    best_d = levenshtein(gt, pred)

    for c in candidates:
        d = levenshtein(gt, c)
        if d < best_d:
            best_d = d
            best = c
            if best_d == 0:
                break

    return best_d, best

def match_plate(gt_text: str, pred_text: str) -> Tuple[bool, Optional[str]]:
    gt = clean_plate(gt_text)
    pred = clean_plate(pred_text)

    if not gt:
        return False, None
    if gt == pred:
        return True, pred

    d, best = pick_best_against_gt(gt, pred)
    return (d == 0), best
