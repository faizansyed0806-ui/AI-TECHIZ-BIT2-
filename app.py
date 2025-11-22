# app.py
"""
Single-file Flask app:
 - Face shape (very simple heuristic) using OpenCV
 - Skin tone (light/medium/deep) using OpenCV (no PIL)
 - TDEE calculation
 - Basic 7-day diet plan

Dependencies: flask, opencv-python
"""

from pathlib import Path
from datetime import datetime
import json
import math
import traceback

from flask import Flask, request, render_template_string, send_file, url_for

import cv2  # OpenCV


# ------------- CONFIG -------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
app.secret_key = "ai-techiz-simple-no-pil"


# =========================================================
# 1) FACE SHAPE (very lightweight heuristic)
# =========================================================

HAAR_FACE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(HAAR_FACE)


def analyze_face_shape(image_path: str, visualize: bool = False) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    if len(faces) == 0:
        return {"error": "No face detected"}

    # largest face
    x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]

    fw, fh = float(w), float(h)
    aspect = fh / (fw + 1e-9)

    # simple approximations
    cheek_w = fw * 0.8
    jaw_w = fw * 0.7
    forehead_w = fw * 0.7

    cheek_ratio = cheek_w / (fw + 1e-9)
    jaw_cheek_ratio = jaw_w / (cheek_w + 1e-9)
    fore_cheek_ratio = forehead_w / (cheek_w + 1e-9)

    # tiny rule set
    if aspect >= 1.25:
        shape = "Oblong"
    elif aspect <= 0.95 and cheek_ratio >= 0.76:
        shape = "Round"
    elif 0.95 < aspect < 1.15 and 0.92 <= jaw_cheek_ratio <= 1.08:
        shape = "Square"
    elif fore_cheek_ratio > 1.05 and jaw_cheek_ratio < 0.9:
        shape = "Heart"
    elif jaw_cheek_ratio < 0.95 and fore_cheek_ratio < 0.95 and cheek_ratio >= 0.7:
        shape = "Diamond"
    else:
        shape = "Oval"

    recs = {
        "Oval": "Most hairstyles suit oval faces; try layers or waves.",
        "Round": "Add height at the crown; avoid heavy straight bangs.",
        "Square": "Soft layers and curls to soften the jawline.",
        "Heart": "Chin-length bobs and side parts to balance a narrow chin.",
        "Diamond": "Add width at forehead and jaw; soft layered styles.",
        "Oblong": "Use bangs and side volume to reduce apparent length.",
    }.get(shape, "Experiment with balanced styles around cheeks and jaw.")

    result = {
        "image_path": image_path,
        "face_bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "aspect_ratio": round(aspect, 3),
        "face_shape": shape,
        "recommendation": recs,
    }

    if visualize:
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            vis,
            shape,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        out_path = str(Path(image_path).with_name(Path(image_path).stem + "_debug.jpg"))
        cv2.imwrite(out_path, vis)
        result["debug_image"] = out_path

    return result


# =========================================================
# 2) SKIN TONE (OpenCV only)
# =========================================================

def analyze_skin_tone(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}

    h, w = img.shape[:2]
    # center patch around lower half of face
    patch_w = max(10, int(w * 0.2))
    patch_h = max(10, int(h * 0.2))
    cx, cy = w // 2, int(h * 0.6)
    x1 = max(0, cx - patch_w // 2)
    y1 = max(0, cy - patch_h // 2)
    x2 = min(w, x1 + patch_w)
    y2 = min(h, y1 + patch_h)
    patch = img[y1:y2, x1:x2]

    if patch.size == 0:
        return {"error": "Patch for skin tone was empty"}

    # mean BGR
    b, g, r = [float(x) for x in cv2.mean(patch)[:3]]
    # convert to brightness
    lum = 0.114 * b + 0.587 * g + 0.299 * r

    if lum >= 190:
        tone = "Fair"
    elif lum >= 160:
        tone = "Light"
    elif lum >= 120:
        tone = "Medium"
    else:
        tone = "Deep"

    if r > b + 10:
        undertone = "Warm"
    elif b > r + 10:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    return {
        "image_path": image_path,
        "mean_bgr": {"b": round(b, 1), "g": round(g, 1), "r": round(r, 1)},
        "luminance": round(lum, 1),
        "skin_tone": tone,
        "undertone": undertone,
    }


# =========================================================
# 3) TDEE + DIET PLAN
# =========================================================

ACTIVITY_MULT = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "very_active": 1.725,
    "extremely_active": 1.9,
}


def calculate_tdee(age: int, sex: str, height_cm: float, weight_kg: float, activity: str) -> float:
    if sex.lower().startswith("m"):
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    mult = ACTIVITY_MULT.get(activity, ACTIVITY_MULT["moderate"])
    return bmr * mult


def generate_weekly_plan(tdee: int, goal: str) -> dict:
    if goal == "weight_loss":
        target = tdee - 500
    elif goal == "muscle_gain":
        target = tdee + 300
    else:
        target = tdee

    target = max(1200, int(round(target)))

    days = []
    for i in range(7):
        days.append({
            "day": i + 1,
            "target_calories": target,
            "meals": [
                {"name": "Breakfast: Oats + fruit + nuts", "calories": int(target * 0.25)},
                {"name": "Lunch: Protein + rice/roti + veggies", "calories": int(target * 0.35)},
                {"name": "Dinner: Light protein + salad", "calories": int(target * 0.3)},
                {"name": "Snack: Yogurt / nuts / fruit", "calories": int(target * 0.1)},
            ],
        })
    return {"tdee": tdee, "goal": goal, "daily_target": target, "days": days}


# =========================================================
# 4) FLASK HELPERS & TEMPLATES
# =========================================================

def allowed_file(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXT


def save_upload(file) -> Path:
    ext = Path(file.filename).suffix
    name = f"img_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}{ext}"
    out = UPLOAD_DIR / name
    file.save(out)
    return out


PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>AI-Techiz Simple (no PIL)</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .box { border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 8px; }
    pre { background:#f7f7f7; padding:10px; border-radius:6px; white-space:pre-wrap; }
    img { max-width:400px; border-radius:8px; margin-top:10px; }
    .error { color:red; }
  </style>
</head>
<body>
  <h1>AI-Techiz: Simple Health & CV Toolkit (no PIL)</h1>

  {% if error %}
  <div class="box error">
    <strong>Error:</strong> {{ error }}
    {% if trace %}<pre>{{ trace }}</pre>{% endif %}
  </div>
  {% endif %}

  <div class="box">
    <h2>1) Face Shape + Skin Tone</h2>
    <form action="/analyze" method="post" enctype="multipart/form-data">
      <input type="file" name="image" required><br><br>
      <label><input type="checkbox" name="visualize" value="1"> Save debug image</label><br><br>
      <button type="submit">Analyze</button>
    </form>

    {% if face %}
      <h3>Face Shape Result</h3>
      <pre>{{ face }}</pre>
    {% endif %}
    {% if skin %}
      <h3>Skin Tone Result</h3>
      <pre>{{ skin }}</pre>
    {% endif %}
    {% if debug %}
      <h3>Debug Image</h3>
      <img src="{{ debug }}">
    {% endif %}
  </div>

  <div class="box">
    <h2>2) TDEE Calculator</h2>
    <form action="/tdee" method="post">
      Age: <input type="number" name="age" required><br>
      Sex:
      <select name="sex"><option value="male">Male</option><option value="female">Female</option></select><br>
      Height (cm): <input type="number" name="height_cm" step="0.1" required><br>
      Weight (kg): <input type="number" name="weight_kg" step="0.1" required><br>
      Activity:
      <select name="activity">
        <option value="sedentary">Sedentary</option>
        <option value="light">Lightly active</option>
        <option value="moderate">Moderately active</option>
        <option value="very_active">Very active</option>
        <option value="extremely_active">Extremely active</option>
      </select><br><br>
      <button type="submit">Calculate TDEE</button>
    </form>

    {% if tdee %}
      <h3>Your TDEE:</h3>
      <pre>{{ tdee }} kcal/day</pre>
    {% endif %}
  </div>

  <div class="box">
    <h2>3) 7-Day Diet Plan</h2>
    <form action="/diet" method="post">
      TDEE: <input type="number" name="tdee" required><br>
      Goal:
      <select name="goal">
        <option value="maintenance">Maintenance</option>
        <option value="weight_loss">Weight Loss</option>
        <option value="muscle_gain">Muscle Gain</option>
      </select><br><br>
      <button type="submit">Generate Plan</button>
    </form>

    {% if diet %}
      <h3>Plan (JSON)</h3>
      <pre>{{ diet }}</pre>
    {% endif %}
  </div>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/analyze", methods=["POST"])
def analyze_route():
    try:
        f = request.files.get("image")
        if not f or f.filename == "" or not allowed_file(f.filename):
            return render_template_string(PAGE, error="Please upload a valid image.")

        img_path = save_upload(f)
        visualize = request.form.get("visualize") == "1"

        face_res = analyze_face_shape(str(img_path), visualize=visualize)
        skin_res = analyze_skin_tone(str(img_path))

        debug_url = None
        if visualize and isinstance(face_res, dict) and "debug_image" in face_res:
            dbg = Path(face_res["debug_image"])
            if dbg.exists():
                dest = UPLOAD_DIR / dbg.name
                if dest.resolve() != dbg.resolve():
                    dest.write_bytes(dbg.read_bytes())
                debug_url = url_for("uploaded_file", filename=dest.name)

        return render_template_string(
            PAGE,
            face=json.dumps(face_res, indent=2),
            skin=json.dumps(skin_res, indent=2),
            debug=debug_url,
        )
    except Exception as e:
        return render_template_string(PAGE, error=str(e), trace=traceback.format_exc())


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    p = UPLOAD_DIR / filename
    if not p.exists():
        return "File not found", 404
    return send_file(p)


@app.route("/tdee", methods=["POST"])
def tdee_route():
    try:
        age = int(request.form["age"])
        sex = request.form["sex"]
        height_cm = float(request.form["height_cm"])
        weight_kg = float(request.form["weight_kg"])
        activity = request.form["activity"]

        tdee_val = calculate_tdee(age, sex, height_cm, weight_kg, activity)
        return render_template_string(PAGE, tdee=int(round(tdee_val)))
    except Exception as e:
        return render_template_string(PAGE, error=str(e), trace=traceback.format_exc())


@app.route("/diet", methods=["POST"])
def diet_route():
    try:
        tdee_val = int(request.form["tdee"])
        goal = request.form["goal"]
        plan = generate_weekly_plan(tdee_val, goal)
        return render_template_string(PAGE, diet=json.dumps(plan, indent=2))
    except Exception as e:
        return render_template_string(PAGE, error=str(e), trace=traceback.format_exc())


if __name__ == "__main__":
    print("Running: http://127.0.0.1:5000")
    app.run(debug=True)
