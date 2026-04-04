import tomllib

class Config:
    def __init__(self, config_path):
        self._config_data = None

        with open(config_path, 'rb') as f:
            self._config_data = tomllib.load(f)

    def get_train_paths(self):
        return self._config_data["train_path"]
    
    def get_test_paths(self):
        return self._config_data["test_path"]
    
    def get_training_param(self, param: str):
        return self._config_data["training_params"][param]
