import tomllib
from pathlib import Path
from utils.paths import Paths

#: Distinguishes "no default given" from a default of None, so an optional key whose legitimate
#: value is None still reads as absent.
_MISSING = object()


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

    def get_training_param(self, param: str, default=_MISSING):
        """Read a key from [training_params], raising KeyError unless a default is given."""
        if default is _MISSING:
            return self._config_data["training_params"][param]
        return self._config_data.get("training_params", {}).get(param, default)

    def get_model_config(self):
        return self._config_data["model"]

    def get_param(self, param_name: str, default=None):
        return self._config_data.get(param_name, default)


def load_config(config_path):
    config_path = Path(config_path)
    if not Path.is_absolute(config_path):
        config_path = Paths.config_dir() / config_path

    with open(config_path, 'rb') as f:
       model_config = tomllib.load(f)
    return model_config
