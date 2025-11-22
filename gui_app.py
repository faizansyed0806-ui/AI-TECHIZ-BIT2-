# gui_app.py
"""
Desktop GUI for:
 - Face shape + skin tone (OpenCV only)
 - TDEE calculation
 - Simple 7-day diet plan

Requirements:
    pip install opencv-python
Built-ins:
    tkinter (no separate install needed)
"""

import math
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tkinter.scrolledtext import ScrolledText

# ----------------- ANALYSIS LOGIC (same as web app, but simplified) -----------------

HAAR_FACE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(HAAR_FACE)


def analyze_face_shape(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return {"error": "No face detected"}

    x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
    fw, fh = float(w), float(h)
    aspect = fh / (fw + 1e-9)

    cheek_w = fw * 0.8
    jaw_w = fw * 0.7
    forehead_w = fw * 0.7

    cheek_ratio = cheek_w / (fw + 1e-9)
    jaw_cheek_ratio = jaw_w / (cheek_w + 1e-9)
    fore_cheek_ratio = forehead_w / (cheek_w + 1e-9)

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

    return {
        "face_shape": shape,
        "aspect_ratio": round(aspect, 3),
        "recommendation": recs,
    }


def analyze_skin_tone(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}

    h, w = img.shape[:2]
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

    b, g, r = [float(x) for x in cv2.mean(patch)[:3]]
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
        "mean_bgr": {"b": round(b, 1), "g": round(g, 1), "r": round(r, 1)},
        "luminance": round(lum, 1),
        "skin_tone": tone,
        "undertone": undertone,
    }


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


# ----------------- TKINTER GUI -----------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI-Techiz: Health & CV Desktop")
        self.geometry("900x700")

        # selected image path
        self.image_path: Path | None = None

        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tabs
        self.tab_cv = ttk.Frame(notebook)
        self.tab_tdee = ttk.Frame(notebook)
        self.tab_diet = ttk.Frame(notebook)

        notebook.add(self.tab_cv, text="Face + Skin")
        notebook.add(self.tab_tdee, text="TDEE")
        notebook.add(self.tab_diet, text="Diet Plan")

        self._build_cv_tab()
        self._build_tdee_tab()
        self._build_diet_tab()

    # ---------- Face & Skin tab ----------
    def _build_cv_tab(self):
        frame_top = ttk.Frame(self.tab_cv)
        frame_top.pack(fill="x", pady=5)

        ttk.Label(frame_top, text="Selected image:").pack(side="left", padx=5)
        self.lbl_image = ttk.Label(frame_top, text="None")
        self.lbl_image.pack(side="left", padx=5)

        ttk.Button(frame_top, text="Browse...",
                   command=self.choose_image).pack(side="right", padx=5)
        ttk.Button(frame_top, text="Analyze",
                   command=self.run_cv_analysis).pack(side="right", padx=5)

        self.txt_cv = ScrolledText(self.tab_cv, height=25)
        self.txt_cv.pack(fill="both", expand=True, padx=5, pady=5)

    def choose_image(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                     ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Choose face image", filetypes=filetypes)
        if path:
            self.image_path = Path(path)
            self.lbl_image.config(text=str(self.image_path.name))

    def run_cv_analysis(self):
        if not self.image_path:
            messagebox.showwarning("No image", "Please choose an image file first.")
            return
        face = analyze_face_shape(str(self.image_path))
        skin = analyze_skin_tone(str(self.image_path))

        self.txt_cv.delete("1.0", tk.END)
        self.txt_cv.insert(tk.END, "FACE SHAPE RESULT\n")
        self.txt_cv.insert(tk.END, json_pretty(face) + "\n\n")
        self.txt_cv.insert(tk.END, "SKIN TONE RESULT\n")
        self.txt_cv.insert(tk.END, json_pretty(skin))

    # ---------- TDEE tab ----------
    def _build_tdee_tab(self):
        frame = ttk.Frame(self.tab_tdee)
        frame.pack(pady=10, padx=10, anchor="nw")

        ttk.Label(frame, text="Age:").grid(row=0, column=0, sticky="w")
        self.ent_age = ttk.Entry(frame, width=10)
        self.ent_age.grid(row=0, column=1, sticky="w")

        ttk.Label(frame, text="Sex:").grid(row=1, column=0, sticky="w")
        self.sex_var = tk.StringVar(value="male")
        ttk.OptionMenu(frame, self.sex_var, "male", "male", "female").grid(row=1, column=1, sticky="w")

        ttk.Label(frame, text="Height (cm):").grid(row=2, column=0, sticky="w")
        self.ent_height = ttk.Entry(frame, width=10)
        self.ent_height.grid(row=2, column=1, sticky="w")

        ttk.Label(frame, text="Weight (kg):").grid(row=3, column=0, sticky="w")
        self.ent_weight = ttk.Entry(frame, width=10)
        self.ent_weight.grid(row=3, column=1, sticky="w")

        ttk.Label(frame, text="Activity:").grid(row=4, column=0, sticky="w")
        self.activity_var = tk.StringVar(value="sedentary")
        ttk.OptionMenu(
            frame, self.activity_var, "sedentary",
            "sedentary", "light", "moderate", "very_active", "extremely_active"
        ).grid(row=4, column=1, sticky="w")

        ttk.Button(frame, text="Calculate TDEE", command=self.run_tdee).grid(row=5, column=0, columnspan=2, pady=10)

        self.lbl_tdee_result = ttk.Label(frame, text="TDEE: - kcal/day")
        self.lbl_tdee_result.grid(row=6, column=0, columnspan=2, sticky="w", pady=5)

    def run_tdee(self):
        try:
            age = int(self.ent_age.get())
            sex = self.sex_var.get()
            height = float(self.ent_height.get())
            weight = float(self.ent_weight.get())
            activity = self.activity_var.get()

            tdee = calculate_tdee(age, sex, height, weight, activity)
            self.lbl_tdee_result.config(text=f"TDEE: {int(round(tdee))} kcal/day")
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numeric values.")

    # ---------- Diet tab ----------
    def _build_diet_tab(self):
        frame_top = ttk.Frame(self.tab_diet)
        frame_top.pack(pady=10, padx=10, anchor="nw")

        ttk.Label(frame_top, text="TDEE (kcal):").grid(row=0, column=0, sticky="w")
        self.ent_diet_tdee = ttk.Entry(frame_top, width=10)
        self.ent_diet_tdee.grid(row=0, column=1, sticky="w")

        ttk.Label(frame_top, text="Goal:").grid(row=1, column=0, sticky="w")
        self.goal_var = tk.StringVar(value="maintenance")
        ttk.OptionMenu(
            frame_top, self.goal_var, "maintenance",
            "maintenance", "weight_loss", "muscle_gain"
        ).grid(row=1, column=1, sticky="w")

        ttk.Button(frame_top, text="Generate Plan",
                   command=self.run_diet).grid(row=2, column=0, columnspan=2, pady=10)

        self.txt_diet = ScrolledText(self.tab_diet, height=25)
        self.txt_diet.pack(fill="both", expand=True, padx=5, pady=5)

    def run_diet(self):
        try:
            tdee = int(self.ent_diet_tdee.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a valid integer TDEE.")
            return

        goal = self.goal_var.get()
        plan = generate_weekly_plan(tdee, goal)

        self.txt_diet.delete("1.0", tk.END)
        self.txt_diet.insert(tk.END, json_pretty(plan))


def json_pretty(obj) -> str:
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    app = App()
    app.mainloop()
