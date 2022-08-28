import csv
import json
import logging
import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from constants import DEVICE
from my_utils import Averager


class Trainer:
    """
    Class to manage the full training pipeline
    """

    def __init__(self, network, train_dataloader, val_dataloader, test_dalaloader, optimizer,
                 features=None,
                 scheduler=None,

                 nb_epochs=10, batch_size=128, reset=False):
        """
        @param network:
        @param dataset_name:
        @param images_dirs:
        @param loss:
        @param optimizer:
        @param nb_epochs:
        @param nb_workers: Number of worker for the dataloader
        """
        self.network = network
        self.train_dataloader=train_dataloader
        self.val_dataloader=val_dataloader
        self.test_dataloader=test_dalaloader
        self.batch_size = batch_size
        self.features=features


        self.optimizer = optimizer
        self.scheduler =scheduler if scheduler else\
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=10,min_lr=1e-5)
        self.mae = torch.nn.L1Loss(reduction="mean")
        self.mse = torch.nn.MSELoss(reduction="mean")
        self.nb_epochs = nb_epochs
        self.experiment_dir = self.network.experiment_dir
        self.model_info_file = os.path.join(self.experiment_dir, "model.json")
        self.model_info_best_file = os.path.join(self.experiment_dir, "model_best.json")

        if reset:
            if os.path.exists(self.experiment_dir):
                shutil.rmtree(self.experiment_dir)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.start_epoch = 0
        if not reset and os.path.exists(self.model_info_file):
            with open(self.model_info_file, "r") as f:
                self.start_epoch = json.load(f)["epoch"] + 1
                self.nb_epochs += self.start_epoch
                logging.info("Resuming from epoch {}".format(self.start_epoch))
        self.network.to(DEVICE)

    def save_model_info(self, infos, best=False):
        json.dump(infos, open(self.model_info_file, 'w'),indent=4)
        if best: json.dump(infos, open(self.model_info_best_file, 'w'),indent=4)

    def fit(self):
        logging.info("Launch training on {}".format(DEVICE))
        self.summary_writer = SummaryWriter(log_dir=self.experiment_dir)
        itr = self.start_epoch * len(self.train_dataloader) * self.batch_size  ##Global counter for steps
        best_mse = 1e20  # infinity
        if os.path.exists(self.model_info_file):
            with open(self.model_info_file, "r") as f:
                model_info = json.load(f)
                lr=model_info["lr"]
                logging.info(f"Setting lr to {lr}")
                for g in self.optimizer.param_groups:
                    g['lr'] = lr

        if os.path.exists(self.model_info_best_file):
            with open(self.model_info_best_file, "r") as f:
                best_model_info = json.load(f)
                best_mse = best_model_info["val_mse"]


        for epoch in range(self.start_epoch, self.nb_epochs):  # Training loop
            self.network.train()
            """"
            0. Initialize loss and other metrics
            """


            running_mae= {"sum":Averager(),"sbp":Averager(),"dbp":Averager()}
            running_mse = {"sum": Averager(), "sbp": Averager(), "dbp": Averager()}
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.nb_epochs}")
            for _, batch in enumerate(pbar):
                """
                Training lopp
                """
                self.optimizer.zero_grad()

                itr += self.batch_size
                features, sbp_true, dbp_true = batch

                """
                1.Forward pass
                """
                sbp_pred, dbp_pred = self.network(features.to(DEVICE))
                """
                2.Loss computation and other metrics
                """
                mse_loss_sbp, mse_loss_dbp = self.mse(sbp_pred, sbp_true.to(DEVICE)) , self.mse(dbp_pred, dbp_true.to(DEVICE))
                mse_loss = mse_loss_dbp + mse_loss_sbp

                mae_loss_sbp, mae_loss_dbp = self.mae(sbp_pred, sbp_true.to(DEVICE)), self.mae(dbp_pred,dbp_true.to(DEVICE))
                mae_loss = mae_loss_dbp + mae_loss_sbp

                """
                                3.Optimizing
                                """
                # mse_loss.backward()
                mae_loss.backward()
                self.optimizer.step()


                running_mse["sum"].send(mse_loss.item())
                running_mse["sbp"]  .send(mse_loss_sbp.item())
                running_mse["dbp"].send(mse_loss_dbp.item())


                running_mae["sum"].send(mae_loss.item())
                running_mae["sbp"]  .send(mae_loss_sbp.item())
                running_mae["dbp"].send(mae_loss_dbp.item())


                pbar.set_description(
                    f"Epoch {epoch + 1}/{self.nb_epochs}.    mae_sbp:{mae_loss_sbp.item()}, mae_dbp:{mae_loss_dbp.item()}")

                """
                4.Writing logs and tensorboard data, loss and other metrics
                """
                self.summary_writer.add_scalar("Train_step/mae", mae_loss.item(), itr)
                self.summary_writer.add_scalar("Train_step/mae_sbp", mae_loss_sbp.item(), itr)
                self.summary_writer.add_scalar("Train_step/mae_dbp", mae_loss_dbp.item(), itr)

            epoch_train_mae, epoch_train_mae_sbp, epoch_train_mae_dbp =[l.value for l in running_mae.values()]
            self.summary_writer.add_scalar("Train_epoch/mae", epoch_train_mae, epoch)
            self.summary_writer.add_scalar("Train_epoch/mae_sbp", epoch_train_mae_sbp, epoch)
            self.summary_writer.add_scalar("Train_epoch/mae_dbp", epoch_train_mae_dbp, epoch)

            epoch_train_mse, epoch_train_mse_sbp, epoch_train_mse_dbp = [l.value for l in running_mse.values()]
            self.summary_writer.add_scalar("Train_epoch/mse", epoch_train_mse, epoch)
            self.summary_writer.add_scalar("Train_epoch/mse_sbp", epoch_train_mse_sbp, epoch)
            self.summary_writer.add_scalar("Train_epoch/mse_dbp", epoch_train_mse_dbp, epoch)



            running_mae,running_mse= self.eval(epoch)
            self.summary_writer.add_scalar("Validation_epoch/mae", running_mae["sum"].value, epoch)
            self.summary_writer.add_scalar("Validation_epoch/mae_sbp", running_mae["sbp"].value, epoch)
            self.summary_writer.add_scalar("Validation_epoch/mae_dbp", running_mae["dbp"].value, epoch)

            self.summary_writer.add_scalar("Validation_epoch/mse", running_mse["sum"].value, epoch)
            self.summary_writer.add_scalar("Validation_epoch/mse_sbp", running_mse["sbp"].value, epoch)
            self.summary_writer.add_scalar("Validation_epoch/mse_dbp", running_mse["dbp"].value, epoch)
            self.summary_writer.add_scalar("Validation_epoch/mse_dbp", running_mse["dbp"].value, epoch)

            self.scheduler.step(running_mae["sum"].value)

            infos = {
                "epoch": epoch,
                "train_mae_sbp": epoch_train_mae_sbp,
                "train_mae_dbp": epoch_train_mae_dbp,
                "train_mae": epoch_train_mae,
                "val_mae_sbp": running_mae["sbp"].value,
                "val_mae_dbp": running_mae["dbp"].value,
                "val_mae": running_mae["sum"].value,
                "train_mse_sbp": epoch_train_mse_sbp,
                "train_mse_dbp": epoch_train_mse_dbp,
                "train_mse": epoch_train_mse,
                "val_mse_sbp": running_mse["sbp"].value,
                "val_mse_dbp": running_mse["dbp"].value,
                "val_mse": running_mse["sum"].value,
                "lr": self.optimizer.param_groups[0]['lr']
            }
            if self.features:
                infos["features"]=self.features



            if running_mse["sbp"].value < best_mse:
                best_mse = running_mse["sbp"].value
                best = True
            else:
                best = False
            self.network.save_state(best=best)
            self.save_model_info(infos, best=best)
            infos_sum={k:infos[k] for k in ["train_mae_sbp","train_mae_dbp","val_mae_sbp","val_mae_dbp"]}
            logging.info(infos_sum)


            ##Updating learning curve file
            learning_curve_file=os.path.join(self.experiment_dir,"learning_curve.csv")
            if not os.path.exists(learning_curve_file):
                with open(learning_curve_file,"w") as input:
                    writer=csv.writer(input)
                    header=list(infos.keys())
                    writer.writerow(header)
            with open(learning_curve_file, "a") as input:
                writer = csv.writer(input)
                row=list(infos.values())
                writer.writerow(row)



    def eval(self, epoch):
        """
        Compute loss and metrics on a validation dataloader
        @return:
        """
        with torch.no_grad():
            self.network.eval()
            running_mae = {"sum": Averager(), "sbp": Averager(), "dbp": Averager()}
            running_mse = {"sum": Averager(), "sbp": Averager(), "dbp": Averager()}
            for _, batch in enumerate(tqdm(self.val_dataloader, desc=f"Validation Epoch {epoch + 1}/{self.nb_epochs}")):
                """
                Training lopp
                """
                """
                1.Forward pass
                """
                features, sbp_true, dbp_true = batch
                sbp_pred, dbp_pred = self.network(features.to(DEVICE))
                """
                2.Loss computation and other metrics
                """
                mse_loss_sbp, mse_loss_dbp = self.mse(sbp_pred, sbp_true.to(DEVICE)) , self.mae(dbp_pred, dbp_true.to(DEVICE))
                mse_loss = mse_loss_dbp + mse_loss_sbp
                running_mse["sum"].send(mse_loss.item())
                running_mse["sbp"].send(mse_loss_sbp.item())
                running_mse["dbp"].send(mse_loss_dbp.item())

                mae_loss_sbp, mae_loss_dbp = self.mae(sbp_pred, sbp_true.to(DEVICE)), self.mae(dbp_pred,
                                                                                               dbp_true.to(DEVICE))
                mae_loss = mae_loss_dbp + mae_loss_sbp
                running_mae["sum"].send(mae_loss.item())
                running_mae["sbp"].send(mae_loss_sbp.item())
                running_mae["dbp"].send(mae_loss_dbp.item())
        return running_mae,running_mse

    def test(self):
        """
        Compute loss and metrics on a validation dataloader
        @return:
        """
        self.network.load_state(best=True)
        test_results = pd.DataFrame({'SBP_true': [],
                                     'DBP_true': [],
                                     'SBP_est': [],
                                     'DBP_est': []})
        with torch.no_grad():
            self.network.eval()
            for _, batch in enumerate(tqdm(self.test_dataloader,"Running test")):
                """
                Training lopp
                """
                """
                1.Forward pass
                """
                features, sbp_true, dbp_true = batch
                sbp_pred, dbp_pred = self.network(features.to(DEVICE))
                sbp_error=torch.abs(sbp_pred.cpu()-sbp_true)
                dbp_error=torch.abs(dbp_pred.cpu()-dbp_true)
                """
                2.Loss computation and other metrics
                """

                batch_results = pd.DataFrame({'SBP_true': sbp_true.numpy().reshape(sbp_true.shape[0]),
                                                'DBP_true': dbp_true.numpy().reshape(dbp_true.shape[0]),
                                                'SBP_est': sbp_pred.cpu().numpy().reshape(sbp_pred.shape[0]),
                                                'DBP_est': dbp_pred.cpu().numpy().reshape(dbp_pred.shape[0]),
                                                'DBP_error':dbp_error.cpu().numpy().reshape(dbp_error.shape[0]),
                                                'SBP_error':sbp_error.cpu().numpy().reshape(sbp_error.shape[0])
                                            },index=None)
                test_results = pd.concat([test_results,batch_results])

        results_file = os.path.join(self.experiment_dir ,'test_results.csv')
        test_results.to_csv(results_file)

        results_file_mae = os.path.join(self.experiment_dir ,'test_results_ae.csv')
        sbp_mae = np.mean(np.abs(test_results["SBP_true"] - test_results["SBP_est"]))
        sbp_aestd = np.std(test_results["SBP_true"] - test_results["SBP_est"])
        dbp_mae = np.mean(np.abs(test_results["DBP_true"] - test_results["DBP_est"]))
        dbp_aestd = np.std(test_results["DBP_true"] - test_results["DBP_est"])

        with open(results_file_mae, "w") as output:
            writer = csv.writer(output)
            writer.writerow(["sbp_mae", "sbp_aestd", "dbp_mae", "dbp_aestd"])
            writer.writerow([sbp_mae, sbp_aestd, dbp_mae, dbp_aestd])



