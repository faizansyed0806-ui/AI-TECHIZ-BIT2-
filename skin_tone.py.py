# diet plan/diet_plan.py
"""
Diet plan generator.

Exports:
    generate_weekly_plan(tdee: int, goal: str, exclusions: list[str]) -> dict
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import math
import random

KCAL_PER_G_PROTEIN = 4
KCAL_PER_G_CARBS = 4
KCAL_PER_G_FAT = 9

MACRO_TEMPLATES = {
    "maintenance": {"carbs": 0.40, "protein": 0.30, "fat": 0.30},
    "weight_loss": {"protein": 0.40, "carbs": 0.30, "fat": 0.30},
    "muscle_gain": {"carbs": 0.40, "protein": 0.30, "fat": 0.30},
}

CALORIE_ADJUSTMENTS = {
    "maintenance": 0,
    "weight_loss": -500,
    "muscle_gain": +300,
}


@dataclass
class Meal:
    id: str
    name: str
    calories: int
    protein_g: float
    carbs_g: float
    fat_g: float
    tags: List[str]


MEALS_DB: List[Meal] = [
    Meal("m1", "Oatmeal with banana & peanut butter", 420, 14, 55, 15, ["vegetarian"]),
    Meal("m2", "Greek yogurt + berries + granola", 380, 20, 45, 10, ["vegetarian"]),
    Meal("m3", "Grilled chicken + quinoa + vegetables", 520, 45, 40, 12, ["dairy_free"]),
    Meal("m4", "Tuna salad + whole grain bread", 480, 38, 36, 16, ["dairy_free"]),
    Meal("m5", "Lentil curry + brown rice", 550, 22, 78, 10, ["vegetarian", "dairy_free"]),
    Meal("m6", "Egg omelette + spinach + toast", 360, 24, 28, 16, []),
    Meal("m7", "Protein shake + banana", 300, 28, 35, 4, []),
    Meal("m8", "Veggie stir-fry with tofu + rice", 500, 20, 70, 15, ["vegetarian"]),
    Meal("m9", "Salmon + sweet potato + greens", 610, 46, 44, 22, ["dairy_free"]),
    Meal("m10", "Chickpea salad with olive oil", 420, 15, 48, 18, ["vegetarian", "dairy_free"]),
]


def rnd(x: float) -> int:
    return int(math.floor(x + 0.5))


def adjust_calories(tdee: int, goal: str) -> int:
    return max(1200, rnd(tdee + CALORIE_ADJUSTMENTS.get(goal, 0)))


def macros_from_percentages(total_cals: int, perc: Dict[str, float]) -> Dict[str, int]:
    pk = total_cals * perc.get("protein", 0)
    ck = total_cals * perc.get("carbs", 0)
    fk = total_cals * perc.get("fat", 0)
    return {
        "protein_g": rnd(pk / KCAL_PER_G_PROTEIN),
        "carbs_g": rnd(ck / KCAL_PER_G_CARBS),
        "fat_g": rnd(fk / KCAL_PER_G_FAT),
    }


def filter_meals(meals: List[Meal], exclusions: Optional[List[str]]) -> List[Meal]:
    if not exclusions:
        return meals[:]

    def ok(meal: Meal) -> bool:
        for ex in exclusions:
            if ex == "vegetarian" and "vegetarian" not in meal.tags:
                return False
            if ex == "dairy_free" and "dairy_free" not in meal.tags:
                return False
        return True

    return [m for m in meals if ok(m)]


def pick_meals_for_day(meals: List[Meal], target: int, variance: float = 0.12) -> List[Meal]:
    low = target * (1 - variance)
    high = target * (1 + variance)
    shuffled = meals[:]
    random.shuffle(shuffled)

    best = None
    best_diff = float("inf")

    for _ in range(200):
        n = random.choice([2, 3, 4])
        combo = random.sample(shuffled, min(n, len(shuffled)))
        total = sum(m.calories for m in combo)
        diff = abs(total - target)
        if low <= total <= high:
            return combo
        if diff < best_diff:
            best_diff = diff
            best = combo

    return best or shuffled[:2]


def day_macros(meals: List[Meal]) -> Dict[str, int]:
    cal = sum(m.calories for m in meals)
    p = sum(m.protein_g for m in meals)
    c = sum(m.carbs_g for m in meals)
    f = sum(m.fat_g for m in meals)
    return {"calories": rnd(cal), "protein_g": rnd(p), "carbs_g": rnd(c), "fat_g": rnd(f)}


def generate_weekly_plan(tdee: int,
                         goal: str = "maintenance",
                         exclusions: Optional[List[str]] = None,
                         meals_db: Optional[List[Meal]] = None) -> Dict[str, object]:
    meals_db = meals_db or MEALS_DB
    usable = filter_meals(meals_db, exclusions)

    target_cals = adjust_calories(tdee, goal)
    macro_perc = MACRO_TEMPLATES.get(goal, MACRO_TEMPLATES["maintenance"])
    macro_targets = macros_from_percentages(target_cals, macro_perc)

    days = []
    for _ in range(7):
        combo = pick_meals_for_day(usable, target_cals)
        macros = day_macros(combo)
        days.append({"meals": [asdict(m) for m in combo], "macros": macros})

    return {
        "input": {
            "tdee": tdee,
            "goal": goal,
            "calorie_target": target_cals,
            "macro_percentages": macro_perc,
            "macro_targets_g_per_day": macro_targets,
            "exclusions": exclusions or [],
        },
        "days": days,
    }


if __name__ == "__main__":
    print(generate_weekly_plan(2400, "weight_loss", ["vegetarian"]))
