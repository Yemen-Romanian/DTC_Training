from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils.config import Config


class AbstractModel(ABC):
    """This is the abstract class for all the trainable models"""

    @abstractmethod
    def train_step(self, batch, device) -> torch.Tensor:
        """Forward pass + loss computation. Returns scalar loss. No backward."""

    @abstractmethod
    def val_step(self, batch, device) -> torch.Tensor:
        """Same as train_step but called under torch.no_grad()."""

    @abstractmethod
    def build_datasets(self, config: Config) -> tuple[Dataset, Dataset, Dataset | None]:
        """Returns (train_dataset, val_dataset, test_dataset).

        val_dataset is used every evaluation_interval epochs to track the best
        model checkpoint. test_dataset is evaluated once at the end of training
        and should be None if no test path is provided in the config.
        """

    @abstractmethod
    def get_module(self) -> nn.Module:
        """Returns the underlying nn.Module for the optimizer and checkpointing."""
