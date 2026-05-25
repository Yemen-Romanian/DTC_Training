import os
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.tracker_evaluation import evaluate_tracker, calculate_average_metrics
from models.abstract_trainable import AbstractTrainable
from utils.config import Config
from utils.experiment_logger import ExperimentLogger


class Trainer:
    """
    Main class for model training.
    Currently, only tracker training is supported.
    """
    def __init__(self, model: AbstractTrainable, config: Config):
        self.model = model
        self.config = config

        train_ds, val_ds, test_ds = model.build_datasets(config)
        batch_size = config.get_training_param('batch_size')
        num_workers = config.get_training_param('data_workers_num')

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self._test_ds_available = test_ds is not None

        self.nn_module = model.get_module()
        lr = config.get_training_param('lr')
        self.optimizer = optim.Adam(self.nn_module.parameters(), lr=lr, weight_decay=1e-4)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        model_id = config.get_model_config()['id']
        experiment_name = f"{model_id}_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = ExperimentLogger(experiment_name)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nn_module.to(self.device)
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")

        self.epoch_num = config.get_training_param('epochs_num')
        self.evaluation_interval = config.get_training_param('evaluation_interval')

    def train(self):
        best_val_iou = 0.0
        best_val_loss = float('inf')

        for epoch in range(self.epoch_num):
            train_loss = self._train_epoch(epoch)
            self.logger.add_scalar('train_loss', train_loss, epoch)

            val_loss = self._val_epoch(epoch)
            self.logger.add_scalar('val_loss', val_loss, epoch)
            self.lr_scheduler.step(val_loss)

            if epoch > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.logger.info("Running evaluation on validation set...")
                avg_results = self._run_evaluation(epoch, self.config.get_val_paths(), 'val')
                if avg_results['iou'] > best_val_iou:
                    best_val_iou = avg_results['iou']
                    self.logger.log_model(self.nn_module)
                    self.logger.info(f"New best model saved with val IoU: {best_val_iou:.8f}")

            self.logger.info(f"Current learning rate: {self.lr_scheduler.get_last_lr()[0]:.6f}")

        if self._test_ds_available:
            self.logger.info("Running final evaluation on test set...")
            self._run_evaluation(self.epoch_num - 1, self.config.get_test_paths(), 'test')

    def _train_epoch(self, epoch) -> float:
        self.nn_module.train()
        total_loss, num_batches = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epoch_num}")

        for batch in pbar:
            loss = self.model.train_step(batch, self.device)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_description(f"Epoch {epoch+1}/{self.epoch_num}, loss: {total_loss/num_batches:.6f}")

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _val_epoch(self, epoch) -> float:
        self.nn_module.eval()
        total_loss, num_batches = 0.0, 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                loss = self.model.val_step(batch, self.device)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.logger.info(f"Epoch {epoch+1}/{self.epoch_num}, Val Loss: {avg_loss:.8f}")
        return avg_loss

    def _run_evaluation(self, epoch, paths_dict, prefix) -> dict:
        self.logger.info(f"Evaluating on {prefix} set...")
        results = evaluate_tracker(self.nn_module.state_dict(), self.config, paths_dict)
        avg_results = calculate_average_metrics(results)

        for metric_name, metric_value in avg_results.items():
            self.logger.add_scalar(f"avg_{prefix}_{metric_name}", metric_value, epoch)

        self.logger.log_dict(results, f"{prefix}_evaluation_results_epoch_{epoch+1}.json")
        self.logger.log_dict(avg_results, f"{prefix}_average_evaluation_results_epoch_{epoch+1}.json")
        self.logger.info(f"{prefix} metrics: {avg_results}")
        return avg_results
