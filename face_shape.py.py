# face_shape.py
"""
Simple face shape analysis using OpenCV Haar cascades.

Exports:
    analyze_face_shape(image_path: str, visualize: bool = False) -> dict
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import cv2
import math

HAAR_FACE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_EYE = cv2.data.haarcascades + "haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(HAAR_FACE)
eye_cascade = cv2.CascadeClassifier(HAAR_EYE)


def detect_face(gray) -> Optional[Tuple[int, int, int, int]]:
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
    return int(x), int(y), int(w), int(h)


def detect_eyes(gray, face_box: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    x, y, w, h = face_box
    roi = gray[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi, 1.1, 4, minSize=(15, 15))
    out = []
    for ex, ey, ew, eh in eyes:
        out.append((x + ex, y + ey, ew, eh))
    out.sort(key=lambda e: e[0])  # left -> right
    return out


def estimate_measures(face_box: Tuple[int, int, int, int]) -> Dict[str, float]:
    x, y, w, h = face_box
    fw, fh = float(w), float(h)
    aspect = fh / (fw + 1e-9)

    cheek_y = y + int(0.30 * h)
    jaw_y = y + int(0.85 * h)
    forehead_y = y + int(0.12 * h)

    cheek_left_x = x + int(0.12 * w)
    cheek_right_x = x + int(0.88 * w)
    jaw_left_x = x + int(0.18 * w)
    jaw_right_x = x + int(0.82 * w)
    forehead_left_x = x + int(0.20 * w)
    forehead_right_x = x + int(0.80 * w)

    cheek_w = float(cheek_right_x - cheek_left_x)
    jaw_w = float(jaw_right_x - jaw_left_x)
    forehead_w = float(forehead_right_x - forehead_left_x)

    return {
        "face_width": fw,
        "face_height": fh,
        "aspect_ratio": aspect,
        "cheekbone_width": cheek_w,
        "jaw_width": jaw_w,
        "forehead_width": forehead_w,
        "cheek_to_face_ratio": cheek_w / (fw + 1e-9),
        "jaw_to_cheek_ratio": jaw_w / (cheek_w + 1e-9),
        "forehead_to_cheek_ratio": forehead_w / (cheek_w + 1e-9),
    }


def classify_shape(m: Dict[str, float]) -> str:
    ar = m["aspect_ratio"]
    jaw_cheek = m["jaw_to_cheek_ratio"]
    fore_cheek = m["forehead_to_cheek_ratio"]
    cheek_ratio = m["cheek_to_face_ratio"]

    if ar >= 1.25:
        return "Oblong"
    if ar <= 0.95 and cheek_ratio >= 0.76:
        return "Round"
    if 0.95 < ar < 1.15 and 0.92 <= jaw_cheek <= 1.08:
        return "Square"
    if fore_cheek > 1.05 and jaw_cheek < 0.90:
        return "Heart"
    if jaw_cheek < 0.95 and fore_cheek < 0.95 and cheek_ratio >= 0.70:
        return "Diamond"
    return "Oval"


def style_recommendations(shape: str) -> Dict[str, str]:
    if shape == "Oval":
        return {
            "hair": "Most hairstyles suit oval faces; try layers or waves.",
            "eyewear": "Square or geometric frames add contrast."
        }
    if shape == "Round":
        return {
            "hair": "Add height on top; long layers or side bangs to elongate the face.",
            "eyewear": "Angular / rectangular frames add definition."
        }
    if shape == "Square":
        return {
            "hair": "Soft layers and curls to soften the jawline.",
            "eyewear": "Round or oval frames soften strong angles."
        }
    if shape == "Heart":
        return {
            "hair": "Chin-length bobs and side-part styles balance a narrower jaw.",
            "eyewear": "Bottom-heavy or aviator frames complement heart shapes."
        }
    if shape == "Diamond":
        return {
            "hair": "Styles adding width at forehead and chin balance high cheekbones.",
            "eyewear": "Cat-eye or rimless frames work well."
        }
    return {
        "hair": "Use bangs and waves to reduce face length.",
        "eyewear": "Tall frames with decorative temples add width."
    }


def analyze_face_shape(image_path: str, visualize: bool = False) -> Dict[str, Any]:
    if not os.path.isfile(image_path):
        return {"error": f"Image not found: {image_path}"}

    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Unable to load image via OpenCV."}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_box = detect_face(gray)
    if face_box is None:
        return {"error": "No face detected."}

    eyes = detect_eyes(gray, face_box)
    measures = estimate_measures(face_box)
    shape = classify_shape(measures)
    recs = style_recommendations(shape)

    result: Dict[str, Any] = {
        "image_path": image_path,
        "face_bbox": {
            "x": int(face_box[0]),
            "y": int(face_box[1]),
            "w": int(face_box[2]),
            "h": int(face_box[3]),
        },
        "eyes_detected": [
            {"x": int(e[0]), "y": int(e[1]), "w": int(e[2]), "h": int(e[3])}
            for e in eyes
        ],
        "measures": {k: float(v) for k, v in measures.items()},
        "face_shape": shape,
        "recommendations": recs,
    }

    if visualize:
        x, y, w, h = face_box
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(vis, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        cv2.putText(
            vis,
            f"Shape: {shape}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        out_path = os.path.splitext(image_path)[0] + "_face_shape_debug.jpg"
        cv2.imwrite(out_path, vis)
        result["debug_image"] = out_path

    return result
