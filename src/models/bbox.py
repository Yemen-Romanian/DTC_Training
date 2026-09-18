from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Axis-aligned box in image pixel coordinates: top-left origin, width/height extent."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
