"""Rectangular image-space regions over a raw, signed density grid.

Each density cell is treated as uniformly distributed mass. Fractional cell
overlap makes rectangular partitions additive without resizing the density.
These are model estimates, not physical density, detections or safety labels.
"""
import math
import re

import numpy as np

MAX_ZONES = 6


def validate_zones(zones):
    if not isinstance(zones, list) or len(zones) > MAX_ZONES:
        raise ValueError(f"zones must be a list with at most {MAX_ZONES} rectangles")
    result, ids = [], set()
    for zone in zones:
        if not isinstance(zone, dict) or set(zone) != {"id", "name", "rect"}:
            raise ValueError("Each zone requires exactly id, name and rect")
        identity, name, rect = zone["id"], zone["name"], zone["rect"]
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", identity) or identity in ids:
            raise ValueError("Zone IDs must be unique letters/numbers/underscore/hyphen identifiers starting with a letter")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 40 or any(ord(c) < 32 for c in name):
            raise ValueError("Zone names must contain 1..40 printable characters")
        if (not isinstance(rect, (list, tuple)) or len(rect) != 4
                or any(isinstance(n, bool) or not isinstance(n, (int, float)) or not 0 <= n <= 1
                       or not math.isfinite(n) for n in rect)):
            raise ValueError("Zone rect must be [left, top, right, bottom] with finite coordinates in [0,1]")
        left, top, right, bottom = rect
        if left >= right or top >= bottom or (right-left)*(bottom-top) < 1e-6:
            raise ValueError("Zone rectangles must have positive area")
        ids.add(identity)
        result.append({"id": identity, "name": name.strip(), "rect": list(map(float, rect))})
    return result


def zone_counts(density, zones):
    """Float64 signed integrals; normalized bounds refer to playback frame edges.

    A full-frame zone equals the original density sum. Overlapping zones count
    shared mass in BOTH zones, so their counts must not be summed as a total.
    """
    if not isinstance(density, np.ndarray) or density.ndim != 2 or not density.size or not np.isfinite(density).all():
        raise ValueError("Expected a finite nonempty 2D raw density map")
    height, width = density.shape
    xs, ys = np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    result = {}
    for zone in validate_zones(zones):
        left, top, right, bottom = zone["rect"]
        wx = np.clip(np.minimum(xs+1, right*width)-np.maximum(xs, left*width), 0, 1)
        wy = np.clip(np.minimum(ys+1, bottom*height)-np.maximum(ys, top*height), 0, 1)
        result[zone["id"]] = float(np.sum(density * wy[:, None] * wx[None, :], dtype=np.float64))
    return result
