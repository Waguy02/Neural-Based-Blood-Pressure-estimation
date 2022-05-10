import argparse
import logging
from torch.optim import Adam
from logger import setup_logger
from networks.network import CustomNetwork
from training.trainer import Trainer
def cli():
    """
   Parsing args
   @return:
   """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reset", "-r", action='store_true', default=False   , help="Start retraining the model from scratch")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.00005, help="Learning rate of Adam optimized")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--model_name", "-n",help="Name of the model. If not specified, it will be automatically generated")
    parser.add_argument("--num_workers", "-w", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="Batch size for training")
    parser.add_argument("--log_level", "-l", type=str, default="INFO")
    parser.add_argument("--autorun_tb","-tb",default=False,action='store_true',help="Autorun tensorboard")
    return parser.parse_args()


def main(args):
    model_name = "base_model" if args.model_name is None else args.model_name##AutoParsing model name
    network=CustomNetwork()
    optimizer = Adam(network.parameters(), lr=args.learning_rate)
    loss=None #"INstancitate loss
    logging.info("Training : "+model_name)
    trainer=Trainer(network,optimizer,loss,model_name,args.num_workers,args.batch_size,args.epochs,args.autorun_tb)
    trainer.fit()
    
    

if __name__ == "__main__":
    args = cli()
    setup_logger(args)
    main(args)
 