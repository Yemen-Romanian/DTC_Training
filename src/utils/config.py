import tomllib
from pathlib import Path

class Config:
    def __init__(self, config_path):
        self._config_data = None

        with open(config_path, 'rb') as f:
            self._config_data = tomllib.load(f)

        current_file_path = Path(__file__)
        self._output_dir = current_file_path.parent.parent.parent / "output"

    def get_train_paths(self):
        return self._config_data["train_path"]
    
    def get_test_paths(self):
        return self._config_data["test_path"]
    
    def get_training_param(self, param: str):
        return self._config_data["training_params"][param]
    
    def get_output_dir_path(self):
        return self._output_dir
