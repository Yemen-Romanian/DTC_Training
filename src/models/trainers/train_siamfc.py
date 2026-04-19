from tqdm import tqdm
import argparse
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.siamfc_dataset import SiamFCDataset
from torchvision.transforms import ToTensor
import os

from models.trackers.siamfc import SiamFCNet
from models.trackers.feature_extractors import AlexNetFeatureExtractor
from models.losses import BalancedLoss
from utils.config import Config
from datasets.mixed_dataset import MixedDataset
from evaluation.tracker_evaluation import evaluate_tracker, calculate_average_metrics
from models.trackers.tracker_factory import create_tracker
from utils.experiment_logger import ExperimentLogger


def train(config, train_loader, test_loader):
    lr = config.get_training_param('lr')
    epoch_num = config.get_training_param('epochs_num')

    logger = ExperimentLogger(f"siamfc_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

    model = SiamFCNet(AlexNetFeatureExtractor())
    loss = BalancedLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info(f"Is CUDA available? {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_test_iou = 0.0
    evaluation_interval = config.get_training_param('evaluation_interval')

    for epoch in range(epoch_num):
        model.train()
        pbar = tqdm(len(train_loader), desc=f"Epoch {epoch+1}/{epoch_num}, loss: {0.0}", total=len(train_loader))
        epoch_loss = 0.0
        num_samples = 0
        
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

        if test_loader is None:
            continue

        model.eval()
        pbar = tqdm(len(test_loader), desc=f"Evaluating on test set", total=len(test_loader))
        with torch.no_grad():
            test_loss = 0.0
            num_samples = 0
            for examplar, search, gt in test_loader:
                examplar, search, gt = examplar.to(device), search.to(device), gt.to(device)
                prediction = model(examplar, search)
                batch_loss = loss(prediction, gt).sum().item()
                test_loss += batch_loss
                num_samples += gt.size(0)
                pbar.update(1)

            test_loss /= num_samples
            logger.add_scalar("test_loss", test_loss, epoch)
            logger.info(f"Epoch {epoch+1}/{epoch_num}, Test Loss: {test_loss:.8f}") 
        pbar.close()

        if epoch % evaluation_interval == 0:
            logger.info(f"Performing evaluation on test set...")
            tracker = create_tracker('siamfc', state_dict=model.state_dict())
            results = evaluate_tracker(tracker, config)
            avg_results = calculate_average_metrics(results)

            for avg_metric_name, avg_metric_value in avg_results.items():
                logger.add_scalar(f"avg_{avg_metric_name}", avg_metric_value, epoch)

            logger.log_dict(results, f"evaluation_results_epoch_{epoch+1}.json")
            logger.log_dict(avg_results, f"average_evaluation_results_epoch_{epoch+1}.json")

            logger.info(f"Value of new metrics: {avg_results}")
            if avg_results["iou"] > best_test_iou:
                best_test_iou = avg_results["iou"]
                logger.log_model(model)
                logger.info(f"New best model saved with test IoU: {best_test_iou:.8f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SiamFC Tracker")
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()

    config = Config(args.config_path)
    train_dataset = MixedDataset(config.get_train_paths())
    test_dataset = MixedDataset(config.get_test_paths())

    train_siamfc_dataset = SiamFCDataset(train_dataset, transform=ToTensor())
    test_siamfc_dataset = SiamFCDataset(test_dataset, transform=ToTensor())

    batch_size = config.get_training_param('batch_size')

    num_workers = os.cpu_count() - 1
    train_loader = DataLoader(train_siamfc_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_siamfc_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train(config, train_loader, test_loader)
