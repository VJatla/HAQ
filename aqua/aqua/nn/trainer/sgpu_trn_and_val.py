import os
import pdb
import sys
import time
from datetime import datetime
import torch
import shutil
import math
from barbar import Bar
from sklearn.metrics import accuracy_score
from torchsummary import summary


class SGPU_TrnAndVal:
    def __init__(self,
                 params,
                 net,
                 optimizer,
                 criterion,
                 trainloader,
                 valloader,
                 cuda_device_id=0):
        """ Trains a model on one GPU.

        Parameters
        ----------
        params: Dict
        A dictionary having training parameters.
        Ex:
        ```
        {
            "max_epochs": 100,
        }
        ```
        net: Custom Network Instance
            Instance of pytorch custom network we will be training
        optimizer: Optimizer instance
        criterion: Loss instance
        trainloader: Instance of Custom DataLoader
            Training data loader instance
        trainloader: Instance of Custom DataLoader
            Validation data loader instance
        cuda_device_id: int, optional
            CUDA device ID. Defaults to 0.
        """
        self.params = params

        # Load network into GPU
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainloader = trainloader
        self.valloader = valloader

        # Load network into GPU device
        self.device = self._get_cuda_device(cuda_device_id)
        self.net.to(self.device)

        # Create checkpoint directory if it does not exist
        self.work_dir = self.params['work_dir']
        if os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)
        print(f"INFO: Creating work directory {self.params['work_dir']}")
        os.makedirs(self.work_dir)

        # Create log json file
        dir_loc = os.path.dirname(self.params['log_pth'])
        self.trn_log = open(f"{dir_loc}/trn_log.json", "w")
        self.val_log = open(f"{dir_loc}/val_log.json", "w")

        # Epochs for which we need to save checkpoints
        self.max_epochs = self.params['max_epochs']
        ckpt_save_interval = self.params['ckpt_save_interval']
        self.ckpt_save_epochs = list(
            range(0, self.max_epochs, ckpt_save_interval))

        # Save model diagram in model.txt
        model_stats = summary(self.net, params['input_shape'])
        summary_str = str(model_stats)
        model_floc = f"{os.path.dirname(self.params['log_pth'])}/model.txt"
        model_f = open(model_floc, "w")
        model_f.write(summary_str)
        model_f.close()

    def _get_cuda_device(self, cuda_device_id):
        """ Sets NVIDIA device with id `cuda_device_id` to
        be used for training.

        Parameters
        ----------
        net: Custom Network Instance
            Instance of pytorch custom network we will be training
        cuda_device_id: int
            CUDA device ID
        """
        if not torch.cuda.is_available():
            raise Exception("ERROR: Cuda is not found in this environment")

        num_cuda_devices = torch.cuda.device_count()
        cuda_devices = list(range(0,
                                  num_cuda_devices))  # cuda index start from 0

        if not (cuda_device_id <= num_cuda_devices - 1):
            raise Exception(
                f"ERROR: Cuda device {cuda_device_id} is not found.\n"
                f"ERROR: Found {num_cuda_devices}("
                f"{cuda_devices}), Cuda devices")

        # If no errors load network onto cuda device
        print("INFO: Network sucessfully loaded into "
              f"CUDA device {cuda_device_id}")
        return f"cuda:{cuda_device_id}"

    def train(self):
        """ Trains a network with specified dataloaser.
        
        Parameters
        ----------

        """
        print(f"INFO: Starting training for {self.max_epochs} epochs")
        now = datetime.now()
        self.trn_log.write(f'{{"mode": "info", "start_time":"{now}" }}')
        self.val_log.write(f'{{"mode": "info", "start_time":"{now}" }}')

        # Epoch loop
        best_val_acc = 0
        best_val_loss = math.inf
        num_epochs_worse = 0
        num_epochs_to_skip = 20
        for epoch in range(self.max_epochs):
            print(
                f"\n*****************Epoch {epoch}**************************")

            # Training
            print("Trn:")
            self.net.train()
            trnloss, trnaccu, trntt = self._train_and_get_loss_and_accuracy()

            # Save model every n epochs
            if epoch in self.ckpt_save_epochs:
                ckpt_loc = f"{self.work_dir}/epoch_{epoch}.pth"
                self._save_model(epoch, trnloss, ckpt_loc)

            # Validation
            print("Val:")
            self.net.eval()
            valloss, valaccu, valtt = self._get_val_loss_and_accuracy()

            # Only after `num_epochs_to_skip` epochs check for epoch with best validation loss
            # If less than `num_epochs_to_skip` epochs best epoch == latest epoch
            if epoch > num_epochs_to_skip:
                if valloss <= best_val_loss:
                    ckpt_loc = f"{self.work_dir}/best.pth"
                    self._save_model(epoch, trnloss, ckpt_loc)
                    best_val_acc = valaccu
                    best_val_loss = valloss
                    best_epoch = epoch
                    num_epochs_worse = 0
                else:
                    num_epochs_worse += 1
            else:
                ckpt_loc = f"{self.work_dir}/best.pth"
                self._save_model(epoch, trnloss, ckpt_loc)
                best_val_acc = valaccu
                best_val_loss = valloss
                best_epoch = epoch
                num_epochs_worse = 0
                

            # Print and save useful information per epoch
            trnstr = (
                f'{{"mode": "trn", "epoch":{epoch}, "acc":{trnaccu}, '
                f'"loss":{trnloss} }}')
            valstr = (
                f'{{"mode": "val", "epoch":{epoch}, "acc":{valaccu}, '
                f'"loss":{valloss} }}')
            best_str = (
                f'{{"mode": "bst", "epoch":{epoch}, "acc":{best_val_acc}, "loss":{best_val_loss},'
                f'"best epoch":{best_epoch} }}')
            self.trn_log.write(f"\n{trnstr}")
            self.val_log.write(f"\n{valstr}")

            # Print to console
            print(f"Trn: {trnstr}")
            print(f"Val: {valstr}")
            print(f"Bst: {best_str}")

        
            # If the validation accuracy keeps decreasing for 5
            # we stop the training.
            if num_epochs_worse >= 5:
                break

        # Recording end time
        now = datetime.now()
        self.trn_log.write(f'\n{{"mode": "info", "end_time":"{now}" }}')
        self.val_log.write(f'\n{{"mode": "info", "end_time":"{now}" }}')

        # Renaming the best model
        ckpt_loc = f"{self.work_dir}/best.pth"
        new_ckpt_loc = f"{self.work_dir}/best_epoch{best_epoch}.pth"
        os.system(f"mv {ckpt_loc} {new_ckpt_loc}")
        
        self.trn_log.close()
        self.val_log.close()

    def _train_and_get_loss_and_accuracy(self):
        """ Trains and returns avrage training loss and accuracy
        """
        # Training Iteration loop
        trngt_lst = []
        trntt_lst = []
        trnloss_lst = []
        trnpred_lst = []
        for idx, data in enumerate(Bar(self.trainloader)):

            # Train for current iteration
            trnloss, trnpred, trngt, tt = self._train_cur_iter(data)

            # Collect training loss and prediction and time taken
            trntt_lst += [tt]
            trngt_lst += trngt
            trnloss_lst += [trnloss]
            trnpred_lst += trnpred

        # Training loss and prediction accuracy per epoch
        trngt_lst = [round(x) for x in trngt_lst]
        trnpred_lst = [round(x) for x in trnpred_lst]
        tot_tt = round(sum(trntt_lst), 5)
        trnaccu = accuracy_score(trngt_lst, trnpred_lst)
        avg_trnloss = round(sum(trnloss_lst) / len(trnloss_lst), 2)

        return avg_trnloss, trnaccu, tot_tt

    def _train_cur_iter(self, data):
        """ Trains f
        """
        # Label and input tensors
        labels, inputs = (data[0].to(self.device, non_blocking=True),
                          data[1].to(self.device, non_blocking=True))
        labels = torch.reshape(labels.float(), (-1, 1))

        # Training Core
        st = time.time()  # Start time
        self.optimizer.zero_grad()  # Zeroing the gradient
        outputs = self.net(inputs)  # Prediction
        loss = self.criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Back propagation
        self.optimizer.step()  # Update the gradients
        et = time.time()  # End time
        tt = round(et - st, 5)  # time taken

        # Collecting information to cpu memory and to list
        ipred = outputs.data.clone()
        ipred = ipred.to("cpu").numpy().flatten().tolist()
        iloss = loss.data.clone()
        iloss = iloss.to("cpu").numpy().tolist()
        ilabels = labels.data.clone()
        ilabels = ilabels.to("cpu").numpy().flatten().tolist()

        return iloss, ipred, ilabels, tt

    def _get_val_loss_and_accuracy(self):
        """ Returns validation accuracy and loss
        """
        with torch.no_grad():
            valloss_lst = []
            valpred_lst = []
            valgt_lst = []
            valtt_lst = []
            for idx, data in enumerate(Bar(self.valloader)):
                valloss, valpred, valgt, valtt = self._eval_cur_iter(data)

                # Collect validation loss and prediction
                valloss_lst += [valloss]
                valpred_lst += valpred
                valtt_lst += [valtt]
                valgt_lst += valgt

            # Validation loss and prediction accuracy
            valgt_lst = [round(x) for x in valgt_lst]
            valpred_lst = [round(x) for x in valpred_lst]
            tot_valtt = round(sum(valtt_lst), 5)
            avg_valloss = round(sum(valloss_lst) / len(valloss_lst), 2)
            valaccu = accuracy_score(valgt_lst, valpred_lst)

        return avg_valloss, valaccu, tot_valtt

    def _eval_cur_iter(self, data):
        """ Tests
        """
        labels, inputs = (data[0].to(self.device, non_blocking=True),
                          data[1].to(self.device, non_blocking=True))
        labels = torch.reshape(labels.float(), (-1, 1))
        ilabels = labels.data.clone()
        ilabels = ilabels.to("cpu").numpy().flatten().tolist()

        st = time.time()
        outputs = self.net(inputs)
        et = time.time()
        ipred = outputs.data.clone()
        ipred = ipred.to("cpu").numpy().flatten().tolist()

        # Loss
        loss = self.criterion(outputs, labels)
        iloss = loss.data.clone()
        iloss = iloss.to("cpu").numpy().tolist()

        # Time taken
        tt = round(et - st, 5)

        return iloss, ipred, ilabels, tt

    def _save_model(self, epoch, trnloss, ckpt_loc):
        """ Saves model as pth file
        """
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': trnloss,
            }, ckpt_loc)
