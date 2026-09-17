from abc import ABC, abstractmethod
from dataclasses import dataclass

from models.bbox import BoundingBox

@dataclass
class SingleObjectTrackResult:
    bbox: BoundingBox
    confidence: float

class SingleObjectTrackerBase(ABC):
    @abstractmethod
    def initialize(self, image, bbox):
        pass

    @abstractmethod
    def track(self, image) -> SingleObjectTrackResult:
        pass

    @abstractmethod
    def to_device(self, device: str):
        pass

