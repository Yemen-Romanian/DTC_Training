from tqdm import tqdm
import argparse
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.siamfc_dataset import SiamFCDataset
import os

from models.losses import BalancedLoss
from utils.config import Config
from datasets.mixed_dataset import MixedDataset
from evaluation.tracker_evaluation import evaluate_tracker, calculate_average_metrics
from models.model_factory import create_net
from utils.experiment_logger import ExperimentLogger


def train(config, train_loader, val_loader):
    lr = config.get_training_param('lr')
    epoch_num = config.get_training_param('epochs_num')
    model_config = config.get_model_config()

    logger = ExperimentLogger(f"siamfc_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

    model = create_net(model_config)
    loss = BalancedLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    logger.info(f"Is CUDA available? {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_iou = 0.0
    evaluation_interval = config.get_training_param('evaluation_interval')

    for epoch in range(epoch_num):
        model.train()
        pbar = tqdm(len(train_loader), desc=f"Epoch {epoch+1}/{epoch_num}, loss: {0.0}", total=len(train_loader))
        epoch_loss = 0.0
        num_samples = 0
        logger.info(f"Current value of learning rate: {lr_scheduler.get_last_lr()[0]:.8f}")
        
        for batch_idx, (examplar, search, gt) in enumerate(train_loader):
            examplar, search, gt = examplar.to(device), search.to(device), gt.to(device)
            prediction = model(examplar, search)

            batch_loss = loss(prediction, gt)
            optimizer.zero_grad()
            batch_loss.mean().backward()
            optimizer.step()

            epoch_loss += batch_loss.sum().item()
            num_samples += gt.size(0)
            pbar.set_description(f"Epoch {epoch+1}/{epoch_num}, loss: {epoch_loss / num_samples:.8f}")
            pbar.update(1)
        logger.add_scalar("train_loss", epoch_loss / num_samples, epoch)
        pbar.close()

        if val_loader is None:
            continue

        model.eval()
        pbar = tqdm(len(val_loader), desc=f"Evaluating on validation set", total=len(val_loader))
        with torch.no_grad():
            val_loss = 0.0
            num_samples = 0
            for examplar, search, gt in val_loader:
                examplar, search, gt = examplar.to(device), search.to(device), gt.to(device)
                prediction = model(examplar, search)
                batch_loss = loss(prediction, gt).sum().item()
                val_loss += batch_loss
                num_samples += gt.size(0)
                pbar.update(1)

            val_loss /= num_samples
            logger.add_scalar("val_loss", val_loss, epoch)
            logger.info(f"Epoch {epoch+1}/{epoch_num}, Validation Loss: {val_loss:.8f}") 
            lr_scheduler.step(val_loss)
        pbar.close()

        if epoch % evaluation_interval == 0:
            logger.info(f"Performing evaluation on validation set...")
            results = evaluate_tracker(model.state_dict(), config, config.get_val_paths())
            avg_results = calculate_average_metrics(results)

            for avg_metric_name, avg_metric_value in avg_results.items():
                logger.add_scalar(f"avg_{avg_metric_name}", avg_metric_value, epoch)

            logger.log_dict(results, f"evaluation_results_epoch_{epoch+1}.json")
            logger.log_dict(avg_results, f"average_evaluation_results_epoch_{epoch+1}.json")

            logger.info(f"Value of new metrics: {avg_results}")
            if avg_results["iou"] > best_val_iou:
                best_val_iou = avg_results["iou"]
                logger.log_model(model)
                logger.info(f"New best model saved with validation IoU: {best_val_iou:.8f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SiamFC Tracker")
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()

    config = Config(args.config_path)
    train_dataset = MixedDataset(config.get_train_paths())
    val_dataset = MixedDataset(config.get_val_paths())

    train_siamfc_dataset = SiamFCDataset(train_dataset, apply_augmentation=True)
    val_siamfc_dataset = SiamFCDataset(val_dataset, apply_augmentation=False)

    batch_size = config.get_training_param('batch_size')

    num_workers = os.cpu_count() - 1
    train_loader = DataLoader(train_siamfc_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_siamfc_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train(config, train_loader, val_loader)
