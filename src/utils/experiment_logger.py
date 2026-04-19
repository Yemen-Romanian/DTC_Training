import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import logging
import json

from pathlib import Path

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ExperimentLogger:
    def __init__(self, experiment_folder_name: str):
        current_file_path = Path(__file__)
        output_dir = current_file_path.parent.parent.parent / "output"
        self.tensorboard_runs_dir = output_dir / "tensorboard_runs" / experiment_folder_name
        self.logging_dir = Path(output_dir) / experiment_folder_name
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.scalars = dict()
        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_runs_dir)

    def info(self, string):
        logging.info(string)

    def warning(self, string):
        logging.warning(string)

    def error(self, string):
        logging.error(string)

    def add_scalar(self, tag, value, iteration: int):
        if tag not in self.scalars:
            self.scalars[tag] = list()
        self.scalars[tag].append(value)
        self.tensorboard_writer.add_scalar(tag, value, iteration)

    def plot_scalars(self):
        image_filename = self.logging_dir / "scalars.png"
        columns_num = 2
        scalars_keys = list(self.scalars.keys())
        rows_num = (len(scalars_keys) + columns_num - 1) // columns_num
        fig, axes = plt.subplots(rows_num, columns_num)

        for i in range(rows_num):
            for j in range(columns_num):
                ax = axes[i, j] if rows_num > 1 else axes[j]
                if i*columns_num + j >= len(scalars_keys):
                    ax.axis('off')
                    continue
                key = scalars_keys[i*columns_num + j]
                plot_data = self.scalars[key]
                ax.plot(plot_data)
                ax.set_title(key)
                ax.set(xlabel='iteration', ylabel='value')

        plt.tight_layout()
        plt.savefig(image_filename)
        plt.close()

    def log_model(self, model):
        torch.save(model.state_dict(), self.logging_dir / "best_model.pth")

    def log_dict(self, metrics_dict, filename):
        with open(self.logging_dir / filename, "w") as f:
            json.dump(metrics_dict, f)
        
    def close(self):
        self.tensorboard_writer.close()
