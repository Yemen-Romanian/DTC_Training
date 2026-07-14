import tomllib
from pathlib import Path
from utils.paths import Paths

class Config:
    def __init__(self, config_path):
        self._config_data = None

        with open(config_path, 'rb') as f:
            self._config_data = tomllib.load(f)

    def get_train_paths(self):
        return self._config_data["train_path"]

    def get_val_paths(self):
        return self._config_data["val_path"]

    def get_test_paths(self) -> dict | None:
        return self._config_data.get("test_path", None)

    def get_training_param(self, param: str):
        return self._config_data["training_params"][param]

    def get_model_config(self):
        return self._config_data["model"]


def load_config(config_path):
    config_path = Path(config_path)
    if not Path.is_absolute(config_path):
        config_path = Paths.config_dir() / config_path

    with open(config_path, 'rb') as f:
       model_config = tomllib.load(f)
    return model_config
