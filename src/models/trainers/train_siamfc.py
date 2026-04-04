import logging
from tqdm import tqdm
import argparse
import datetime
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.siamfc_dataset import SiamFCDataset
from torchvision.transforms import ToTensor

from models.trackers.siamfc import SiamFCNet
from models.trackers.feature_extractors import AlexNetFeatureExtractor
from models.losses import BalancedLoss
from utils.config import Config
from datasets.mixed_dataset import MixedDataset

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def train(train_loader, test_loader, output_path, lr, epoch_num):
    experiment_dir = Path(output_path) / "training" / f"siamfc_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    model = SiamFCNet(AlexNetFeatureExtractor())
    loss = BalancedLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logging.info(f"Is CUDA available? {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_test_loss = float('inf')

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
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), experiment_dir / "best_siamfc.pth")
                logging.debug(f"New best model saved with test loss: {best_test_loss:.8f}")
            logging.info(f"Epoch {epoch+1}/{epoch_num}, Test Loss: {test_loss:.8f}") 
        pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SiamFC Tracker")
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()

    config = Config(args.config_path)
    train_dataset = MixedDataset(config.get_train_paths())
    test_dataset = MixedDataset(config.get_test_paths())
    output_path = config.get_output_dir_path()

    train_siamfc_dataset = SiamFCDataset(train_dataset, transform=ToTensor())
    test_siamfc_dataset = SiamFCDataset(test_dataset, transform=ToTensor())

    batch_size = config.get_training_param('batch_size')
    lr = config.get_training_param('lr')
    epochs_num = config.get_training_param('epochs_num')

    train_loader = DataLoader(train_siamfc_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_siamfc_dataset, batch_size=batch_size, shuffle=False)

    train(train_loader, test_loader, output_path, lr, epochs_num)
