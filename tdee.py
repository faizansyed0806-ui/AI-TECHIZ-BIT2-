# tdee/tdee.py
"""
TDEE calculator.

Exports:
    calculate_tdee(age, sex, height_cm, weight_kg, activity_level) -> float
"""

from typing import Literal

ActivityLevel = Literal[
    "sedentary",
    "light",
    "moderate",
    "very_active",
    "extremely_active",
]


ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "very_active": 1.725,
    "extremely_active": 1.9,
}


def mifflin_st_jeor_bmr(age: int, sex: str, height_cm: float, weight_kg: float) -> float:
    if sex.lower().startswith("m"):
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


def calculate_tdee(age: int,
                   sex: str,
                   height_cm: float,
                   weight_kg: float,
                   activity_level: ActivityLevel = "moderate") -> float:
    bmr = mifflin_st_jeor_bmr(age, sex, height_cm, weight_kg)
    mult = ACTIVITY_MULTIPLIERS.get(activity_level, ACTIVITY_MULTIPLIERS["moderate"])
    return bmr * mult


if __name__ == "__main__":
    print(calculate_tdee(22, "male", 175, 70, "moderate"))
