import argparse
import os

import torch.optim
from constants import EXPERIMENTS_DIR, FEATURES_DATA_DIR
from dataset.dataset_features import DatasetType, DatasetFeatures
from logger import setup_logger
from my_utils import create_dataloader
from networks.mlp import MLP
from training.trainer import TrainerMLP
def cli():
    """
   Parsing args
   @return:
   """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reset", "-r", action='store_true', default=False, help="Start retraining the model from scratch")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.1,help="Learning rate of Adam optimized")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--model_name", "-n",help="Name of the model. If not specified, it will be automatically generated")
    parser.add_argument("--num_workers", "-w", type=int, default=2, help="Number of workers for data loading")
    parser.add_argument("--batch_size", "-bs", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--log_level", "-l", type=str, default="INFO")

    return parser.parse_args()


def main(args):
    model_name = "mlp_"+os.path.basename(FEATURES_DATA_DIR) if args.model_name is None else args.model_name##AutoParsing model name
    experiment_dir=os.path.join(EXPERIMENTS_DIR,model_name)
    network=MLP(experiment_dir=experiment_dir, reset=args.reset)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)

    train_dataloader = create_dataloader(DatasetFeatures(DatasetType.TRAIN), batch_size=args.batch_size,num_workers=args.num_workers, shuffle=True)
    val_dataloader = create_dataloader(DatasetFeatures(DatasetType.VAL), batch_size=args.batch_size,num_workers=args.num_workers)
    test_dataloader = create_dataloader(DatasetFeatures(DatasetType.TEST), batch_size=args.batch_size,num_workers=args.num_workers)

    trainer = TrainerMLP(network, train_dataloader, val_dataloader, test_dataloader, optimizer=optimizer,nb_epochs=args.epochs, batch_size=args.batch_size, reset=args.reset)
    trainer.fit()
    trainer.test()


if __name__ == "__main__":
    args = cli()
    setup_logger(args)
    main(args)
 