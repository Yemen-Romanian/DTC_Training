from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from models.bbox import BoundingBox

@dataclass
class DetectionResult:
    bbox: BoundingBox = field(default_factory=BoundingBox)
    confidence: float = 0.0
    class_name: str = ""

class AbstractDetector(ABC):
    @abstractmethod
    def detect(self, image) -> List[DetectionResult]:
        """Detect objects in a single frame.

        Args:
            image: RGB uint8 frame of shape (H, W, 3), as yielded by VideoSource
                and as accepted by SingleObjectTrackerBase.track().

        Returns:
            One DetectionResult per detection; an empty list if there are none.
        """
        pass
