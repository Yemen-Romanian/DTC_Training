from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

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

