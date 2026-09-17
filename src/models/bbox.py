from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Axis-aligned box in image pixel coordinates: top-left origin, width/height extent."""
    x: int
    y: int
    width: int
    height: int
