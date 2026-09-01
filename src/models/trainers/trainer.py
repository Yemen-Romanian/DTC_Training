import os
import datetime
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from datasets.dataset_factory import create_dataset
from evaluation.tracker_evaluation import evaluate_tracker, calculate_average_metrics
from models.abstract_trainable import AbstractTrainable
from models.trackers.tracker_factory import resolve_state_dict
from utils.config import Config
from utils.experiment_logger import ExperimentLogger
from utils.tools_MLFlower import MLFlower
from utils.mlflow_logging import save_training_run


# Validation is a few dozen batches and runs GPU-bound, so it finishes in roughly the
# time a full-sized worker pool takes to spawn. A small pool serves it just as fast and
# halves the per-epoch spawn cost.
VAL_DATA_WORKERS = 4

# evaluate_tracker runs its videos in its own ProcessPoolExecutor, which defaults to
# os.cpu_count() - 1. The train loader's pool is persistent and therefore still resident
# during evaluation, so an uncapped pool would put ~2 x data_workers_num processes on the
# machine at once. On Windows each of those is a full interpreter and the commit charge
# does not survive it.
EVAL_WORKERS = 6


def init_data_worker(worker_id: int):
    """Stop each DataLoader worker from spawning its own thread pools.

    OpenCV and torch both default to a pool sized to the machine's cores, so N workers
    claim N x cores threads and spend most of their time in pool synchronization rather
    than decoding. The per-sample ops here are far too small to benefit from intra-op
    parallelism anyway. Must stay a module-level function so it pickles under Windows
    spawn.
    """
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


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

        # The train pool is persistent so it is spawned once per run instead of once per
        # epoch - on Windows each worker is a fresh interpreter importing torch, which
        # dominated the per-epoch startup. The consequence is that it stays resident while
        # the val pool and the evaluation pool exist, so both of those are kept small
        # (VAL_DATA_WORKERS / EVAL_WORKERS) to bound the concurrent process count. Raising
        # any of the three without checking the commit charge reintroduces the
        # "paging file is too small" failure.
        loader_kwargs = dict(batch_size=batch_size)
        if num_workers > 0:
            loader_kwargs.update(worker_init_fn=init_data_worker)
        val_workers = min(VAL_DATA_WORKERS, num_workers)

        # persistent_workers=True removes the per-epoch respawn of this pool (worth ~90s
        # an epoch on Windows) but keeps it resident alongside the val and evaluation
        # pools. Measured at a 45.8 GiB commit limit that peaks at 23 processes and a
        # 0.3 GiB commit floor - it survives, but with no margin. Enable it only with a
        # larger (ideally fixed-size) page file; see EVAL_WORKERS above.
        self.train_loader = DataLoader(train_ds, shuffle=True, num_workers=num_workers,
                                       persistent_workers=True, **loader_kwargs)
        self.val_loader = DataLoader(val_ds, shuffle=False,
                                     num_workers=val_workers, **loader_kwargs)
        self._test_ds_available = test_ds is not None

        self.nn_module = model.get_module()
        lr = config.get_training_param('lr')
        self.optimizer = optim.Adam(self.nn_module.parameters(), lr=lr, weight_decay=1e-4)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        model_id = config.get_model_config()['id']
        run_name = config.get_param('run_name') or f"{model_id}_training"
        self.run_dir_name = f"{run_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = ExperimentLogger(self.run_dir_name)

        self.logger.log_dict(config.as_dict(), "effective_config.json")
        for override in config.get_overrides():
            self.logger.info(f"Config override applied: {override}")

        mlflow_logging = config.get_param("mlflow_logging", False)
        self.mlflower = MLFlower(
            host = os.environ['MLFLOW_HOST'],
            port = os.environ['MLFLOW_PORT'],
            username_mlflow = os.environ.get('MLFLOW_TRACKING_USERNAME'),
            password_mlflow = os.environ.get('MLFLOW_TRACKING_PASSWORD')
        ) if mlflow_logging else None

        if mlflow_logging:
            self.logger.info(f"MLFlower available: {self.mlflower.is_available}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nn_module.to(self.device)
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")

        self._load_pretrained_weights(config.get_model_config())

        self.epoch_num = config.get_training_param('epochs_num')
        self.evaluation_interval = config.get_training_param('evaluation_interval')
        self.model_config = config.get_model_config()

        self.eval_val_videos = self._parse_eval_videos(config.get_val_paths())
        test_paths = config.get_test_paths()
        self.eval_test_videos = self._parse_eval_videos(test_paths) if test_paths else None

        self.best_model_path = None
        self.train_loss_history = list()
        self.val_loss_history = list()

    def _load_pretrained_weights(self, model_config: dict):
        """Start training from the checkpoint listed as `weights` in the model config, if any.

        Path resolution is shared with the tracker factory: an absolute path is used as is,
        a bare file name is looked up in output/model_weights. Without a `weights` entry the
        module keeps its random initialization, i.e. training starts from scratch.
        """
        weights_path = resolve_state_dict(model_config)
        if weights_path is None:
            self.logger.info("No pre-trained weights in the model config, training from scratch.")
            return

        self.nn_module.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.logger.info(f"Loaded pre-trained weights from {weights_path}")

    def train(self):
        best_val_loss = float('inf')
        best_val_metrics = {}

        for epoch in range(self.epoch_num):
            train_loss = self._train_epoch(epoch)
            self.logger.add_scalar('train_loss', train_loss, epoch)
            self.train_loss_history.append(train_loss)

            val_loss = self._val_epoch(epoch)
            self.logger.add_scalar('val_loss', val_loss, epoch)
            self.val_loss_history.append(val_loss)
            self.lr_scheduler.step(val_loss)
            self._log_branch_weights(epoch)

            if epoch > 0 and (val_loss < best_val_loss or abs(val_loss - best_val_loss) < 0.05):
                best_val_loss = min(best_val_loss, val_loss)
                self.logger.info("Running evaluation on validation set...")
                avg_results = self._run_evaluation(epoch, self.eval_val_videos, 'val')

                if (not best_val_metrics) or (avg_results['iou'] > best_val_metrics['iou'] or avg_results['iog'] > best_val_metrics['iog']):
                    best_val_metrics = avg_results
                    checkpoint_name = "_".join([f"{metric_name}_{avg_results[metric_name]:.4f}" for metric_name in avg_results])
                    self.logger.log_model(self.nn_module, name=checkpoint_name)
                    self.best_model_path = self.logger.logging_dir / f"{checkpoint_name}.pth"
                    self.logger.info(f"New model saved with val loss: {val_loss:.8f}. Current best val loss: {best_val_loss:.8f}.")

            self.logger.info(f"Current learning rate: {self.lr_scheduler.get_last_lr()[0]:.6f}")

        # Restore the best-by-validation checkpoint so the final test evaluation and the
        # logged model both reflect the best weights, not the last epoch's (which the
        # in-memory module has drifted to by the end of training).
        if self.best_model_path is not None:
            self.logger.info(f"Loading best checkpoint before final evaluation: {self.best_model_path}")
            self.nn_module.load_state_dict(torch.load(self.best_model_path, map_location=self.device))

        test_metrics = {}
        if self._test_ds_available:
            self.logger.info("Running final evaluation on test set...")
            test_metrics = self._run_evaluation(self.epoch_num - 1, self.eval_test_videos, 'test')

        if self.mlflower is not None and self.mlflower.is_available:
            # add epoch metrics in the future
            history = {"train_loss": self.train_loss_history, "val_loss": self.val_loss_history}
            self.logger.info("Pushing results to MLflow...")
            run_id = save_training_run(
                self.mlflower, self.config, self.nn_module,
                best_val_metrics, test_metrics, history, self.best_model_path,
                run_dir=self.run_dir_name
            )
            self.logger.info(f"Results pushed to MLflow (run_id={run_id}).")

    def _branch_weights(self):
        """Normalized backbone fusion weights, or None if this backbone has no branches.

        Only multi-tap backbones (MobileNetV3MultiLayerFeatureExtractor) expose these. They are
        worth watching: they say how much the network actually leans on each tap, so a run whose
        weights collapse onto one branch has answered the multi-layer question negatively long
        before the metrics do.
        """
        backbone = getattr(self.nn_module, 'backbone', None)
        if backbone is None or not hasattr(backbone, 'branch_weights'):
            return None
        with torch.no_grad():
            weights = backbone.branch_weights().detach().cpu().tolist()
        taps = getattr(backbone, 'taps', tuple(range(len(weights))))
        return list(zip(taps, weights))

    def _log_branch_weights(self, epoch):
        """Record the fusion weights for the epoch, to the log and to TensorBoard."""
        branches = self._branch_weights()
        if branches is None:
            return
        pretty = ", ".join(f"features[:{tap}]={weight:.4f}" for tap, weight in branches)
        self.logger.info(f"Backbone branch weights: {pretty}")
        for tap, weight in branches:
            self.logger.add_scalar(f"branch_weight/features_{tap}", weight, epoch)

    def _train_epoch(self, epoch) -> float:
        self.nn_module.train()
        total_loss, num_batches = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epoch_num}")
        has_branches = self._branch_weights() is not None

        for batch in pbar:
            loss = self.model.train_step(batch, self.device)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_description(f"Epoch {epoch+1}/{self.epoch_num}, loss: {total_loss/num_batches:.6f}")
            # Throttled: reading the weights forces a device sync, and they drift slowly.
            if has_branches and num_batches % 20 == 0:
                pbar.set_postfix_str(
                    "w " + "/".join(f"{weight:.3f}" for _, weight in self._branch_weights())
                )

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

    @staticmethod
    def _parse_eval_videos(paths_dict: dict) -> dict:
        return {label: create_dataset(label, path).parse() for label, path in paths_dict.items()}

    def _run_evaluation(self, epoch, videos, prefix) -> dict:
        self.logger.info(f"Evaluating on {prefix} set...")
        # Uncapped (os.cpu_count() - 1) is fine only because the train loader is not
        # persistent, so its pool is already torn down by the time this runs. Cap this to
        # EVAL_WORKERS if persistent_workers is ever enabled on the train loader.
        results = evaluate_tracker(self.model_config, videos, state_dict=self.nn_module.state_dict())
        avg_results = calculate_average_metrics(results)

        for metric_name, metric_value in avg_results.items():
            self.logger.add_scalar(f"avg_{prefix}_{metric_name}", metric_value, epoch)

        self.logger.log_dict(results, f"{prefix}_evaluation_results_epoch_{epoch+1}.json")
        self.logger.log_dict(avg_results, f"{prefix}_average_evaluation_results_epoch_{epoch+1}.json")
        self.logger.info(f"{prefix} metrics: {avg_results}")
        return avg_results
