import logging
from tqdm import tqdm
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.siamfc_dataset import SiamFCDataset
from torchvision.transforms import ToTensor

from models.trackers.siamfc import SiamFCNet
from models.trackers.feature_extractors import AlexNetFeatureExtractor
from models.losses import BalancedLoss

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def train(dataset_path, batch_size, lr, epoch_num):
    dataset = SiamFCDataset(dataset_path, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SiamFCNet(AlexNetFeatureExtractor())
    loss = BalancedLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logging.info(f"Is CUDA available? {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epoch_num):
        pbar = tqdm(len(dataloader), desc=f"Epoch {epoch+1}/{epoch_num}, loss: {0.0}", total=len(dataloader))
        epoch_loss = 0.0
        num_samples = 0
        
        for batch_idx, (examplar, search, gt) in enumerate(dataloader):
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

        torch.save(model.state_dict(), "siamfc.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SiamFC Tracker")
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch_num', type=int, default=10)
    args = parser.parse_args()
    train(args.dataset_path, args.batch_size, args.lr, args.epoch_num)