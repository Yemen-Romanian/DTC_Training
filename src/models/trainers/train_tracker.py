import argparse

from models.trainers.trainer import Trainer
from models.trainers.trainables_factory import create_trainable
from utils.config import Config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main script for tracker training from scratch")
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()
    config = Config(args.config_path)
    trainable_model = create_trainable(config.get_model_config())
    trainer = Trainer(trainable_model, config)
    trainer.train()
