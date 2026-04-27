from pathlib import Path

class Paths:
    _current_file_path = Path(__file__)

    @classmethod
    def output_dir(cls):
        return cls._current_file_path.parent.parent.parent / "output"
    
    @classmethod
    def tracking_output_dir(cls):
        return cls.output_dir / "tracking"
    
    @classmethod
    def training_output_dir(cls):
        return cls.output_dir() / "training"
    
    @classmethod
    def model_weights_dir(cls):
        return cls.output_dir() / "model_weights"
