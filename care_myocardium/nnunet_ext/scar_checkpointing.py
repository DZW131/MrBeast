from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def scar_dice_from_pseudo_dice(pseudo_dice: Sequence[float], scar_index: int = -1) -> float:
    values = list(pseudo_dice)
    if not values:
        return math.nan
    index = scar_index if scar_index >= 0 else len(values) + scar_index
    if index < 0 or index >= len(values):
        return math.nan
    return float(values[index])


def best_scar_from_history(history: Iterable[Sequence[float]], scar_index: int = -1) -> float | None:
    best: float | None = None
    for pseudo_dice in history:
        value = scar_dice_from_pseudo_dice(pseudo_dice, scar_index)
        if should_update_best_scar(value, best):
            best = value
    return best


def should_update_best_scar(current: float, previous_best: float | None) -> bool:
    if not math.isfinite(float(current)):
        return False
    if previous_best is None or not math.isfinite(float(previous_best)):
        return True
    return float(current) > float(previous_best)
