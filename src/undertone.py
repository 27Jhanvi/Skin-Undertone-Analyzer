from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import cv2

@dataclass
class UndertoneResult:
    label: str
    confidence: float
    debug: Dict

def lab_stats_from_rgb_mask(image_rgb: np.ndarray, mask: np.ndarray, min_sat: int = 35) -> Optional[Dict]:
    """
    Convert RGB -> LAB and extract robust median a*, b* over masked pixels.
    We filter out low-saturation pixels to reduce background/lighting noise.
    Returns None if no valid pixels.
    """
    if image_rgb is None or mask is None:
        return None

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]

    valid = (mask == 255) & (sat >= int(min_sat))
    if valid.sum() < 400:  # too few pixels = unreliable
        return None

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[..., 1][valid] - 128.0  # a* (red-green), + = red/pink, - = green
    b = lab[..., 2][valid] - 128.0  # b* (yellow-blue), + = yellow, - = blue

    med_a = float(np.median(a))
    med_b = float(np.median(b))

    # Angle of (a*, b*) vector (0°=+a axis, 90°=+b axis)
    angle = float(np.degrees(np.arctan2(med_b, med_a)))

    return {
        "med_a": med_a,
        "med_b": med_b,
        "angle": angle,
        "n_pixels": int(valid.sum()),
    }

def _base_class_from_ab(med_a: float, med_b: float, neutral_band: float = 2.5) -> str:
    """
    Basic rules:
    - Warm: b* significantly positive (yellow)
    - Cool: b* significantly negative (blue-ish)
    - Neutral: close to 0 within tolerance
    """
    if med_b > neutral_band:
        return "Warm"
    if med_b < -neutral_band:
        return "Cool"
    return "Neutral"

def _olive_flag(med_a: float, med_b: float) -> bool:
    """
    Olive often shows:
    - slightly lower a* (less red) and positive b* (yellow)
    This is heuristic — keep it conservative.
    """
    return (med_b >= 8.0) and (med_a <= 6.0)

def classify_from_two_conditions(
    nat_stats: Optional[Dict],
    ind_stats: Optional[Dict],
    neutral_band: float = 2.5
) -> UndertoneResult:
    """
    Combine natural + indoor when available. Use medians of each.
    Produces label + confidence + debug.
    """
    if nat_stats is None and ind_stats is None:
        return UndertoneResult("Unknown", 0.0, {"reason": "no_valid_pixels"})

    # pick mode
    if nat_stats is not None and ind_stats is not None:
        med_a = (nat_stats["med_a"] + ind_stats["med_a"]) / 2.0
        med_b = (nat_stats["med_b"] + ind_stats["med_b"]) / 2.0
        mode = "dual_avg"
        nat_med_a, nat_med_b = nat_stats["med_a"], nat_stats["med_b"]
        ind_med_a, ind_med_b = ind_stats["med_a"], ind_stats["med_b"]
        n_pixels = nat_stats["n_pixels"] + ind_stats["n_pixels"]
    else:
        chosen = nat_stats if nat_stats is not None else ind_stats
        med_a, med_b = chosen["med_a"], chosen["med_b"]
        mode = "single_natural" if nat_stats is not None else "single_indoor"
        nat_med_a = nat_stats["med_a"] if nat_stats else None
        nat_med_b = nat_stats["med_b"] if nat_stats else None
        ind_med_a = ind_stats["med_a"] if ind_stats else None
        ind_med_b = ind_stats["med_b"] if ind_stats else None
        n_pixels = chosen["n_pixels"]

    base = _base_class_from_ab(med_a, med_b, neutral_band=neutral_band)

    # confidence based on distance from neutral band (b* magnitude mostly)
    margin = abs(med_b) - neutral_band
    conf = float(1.0 / (1.0 + np.exp(-0.45 * margin)))  # smooth 0-1
    conf = max(0.15, min(conf, 0.95))  # clamp

    olive = _olive_flag(med_a, med_b)
    label = base
    if olive and base != "Cool":
        label = f"{base}-Olive"

    debug = {
        "mode": mode,
        "med_a": round(med_a, 2),
        "med_b": round(med_b, 2),
        "neutral_band": neutral_band,
        "olive_flag": olive,
        "base": base,
        "margin": round(margin, 2),
        "n_pixels": n_pixels,
        "nat_med_a": None if nat_med_a is None else round(nat_med_a, 2),
        "nat_med_b": None if nat_med_b is None else round(nat_med_b, 2),
        "ind_med_a": None if ind_med_a is None else round(ind_med_a, 2),
        "ind_med_b": None if ind_med_b is None else round(ind_med_b, 2),
    }

    return UndertoneResult(label=label, confidence=conf, debug=debug)
