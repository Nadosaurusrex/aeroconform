"""Airspace definitions for European FIR/TMA boundaries.

Provides bounding boxes for key Flight Information Regions used
in AeroConform data collection and monitoring.
"""

from __future__ import annotations

# FIR bounding boxes: (min_lat, max_lat, min_lon, max_lon)
FIR_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "EGTT": (49.0, 61.0, -12.0, 2.0),    # London FIR
    "LIMM": (43.5, 47.0, 6.5, 14.0),     # Milan FIR
    "LFFF": (45.0, 51.5, -5.5, 8.5),     # Paris FIR
}

FIR_NAMES: dict[str, str] = {
    "EGTT": "London FIR",
    "LIMM": "Milan FIR",
    "LFFF": "Paris FIR",
}


def get_fir_bbox(fir_code: str) -> tuple[float, float, float, float]:
    """Get the bounding box for a Flight Information Region.

    Args:
        fir_code: ICAO FIR code (e.g. 'EGTT', 'LIMM', 'LFFF').

    Returns:
        Bounding box as (min_lat, max_lat, min_lon, max_lon).

    Raises:
        ValueError: If the FIR code is not recognized.
    """
    code_upper = fir_code.upper()
    if code_upper not in FIR_BBOXES:
        available = ", ".join(sorted(FIR_BBOXES.keys()))
        raise ValueError(f"Unknown FIR code: {fir_code!r}. Available: {available}")
    return FIR_BBOXES[code_upper]


def get_fir_name(fir_code: str) -> str:
    """Get the human-readable name for a Flight Information Region.

    Args:
        fir_code: ICAO FIR code (e.g. 'EGTT', 'LIMM', 'LFFF').

    Returns:
        Human-readable FIR name.

    Raises:
        ValueError: If the FIR code is not recognized.
    """
    code_upper = fir_code.upper()
    if code_upper not in FIR_NAMES:
        available = ", ".join(sorted(FIR_NAMES.keys()))
        raise ValueError(f"Unknown FIR code: {fir_code!r}. Available: {available}")
    return FIR_NAMES[code_upper]


def list_firs() -> list[dict[str, str | tuple[float, float, float, float]]]:
    """List all available Flight Information Regions.

    Returns:
        List of dicts with 'code', 'name', and 'bbox' keys.
    """
    return [
        {"code": code, "name": FIR_NAMES[code], "bbox": bbox}
        for code, bbox in sorted(FIR_BBOXES.items())
    ]
