from abc import ABC, abstractmethod


class TrackerBase(ABC):
    @abstractmethod
    def initialize(self, image, bbox):
        pass

    @abstractmethod
    def track(self, image):
        pass
