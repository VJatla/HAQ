import os
import pdb
import sys
import time
import json
import math
import torch
import shutil
import pandas as pd
from barbar import Bar
from datetime import datetime
from torchsummary import summary
from sklearn.metrics import accuracy_score


class SGPU_Val:
    def __init__(self, workdir, net, vloader, max_epochs, criterion, eskip=1):
        """ Validates training checkpoints.

        Parameters
        ----------
        workdir: str
            Training directory having checkpoints
        net: Custom Network Instance
            Instance of pytorch custom network we will be 
            training
        vloader: Instance of Custom DataLoader
            Validation data loader instance
        max_epochs: int
            Maximum number of epochs for which the current
            network is trained.
        criterion: Loss instance
        eskip: int, optional
            Epoch skip interval between validations.
            Defaults to 1, performs validation for every
            epoch.
        
        Note
        ----
        1. Please use `CUDA_VISIBLE_DEVICES=1` to select 
           GPU. This code assumes to use GPU 0 always.
        2. The checkpoint files are assumed to be named
           `epoch_<#>.pth`.
        3. It also checks for training log file in working
           directory. It is assumed to be named trn_log.json
           
        """
        self.workdir = workdir
        self.net = net
        self.vloader = vloader
        self.max_epochs = max_epochs
        self.eskip = eskip
        self.criterion = criterion

        # Load training log df from json file
        self._trn_df = self._load_trn_json()

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

    def validate(self, log_path):
        """ Validation is performed here
        """
        # Open log file to write
        log = open(log_path, "w")
        now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        log.write(f'{{"mode": "info", "start_time":"{now}" }}')

        # Check if required checkpoint files are present
        epoch_lst = list(range(0, self.max_epochs, self.eskip))
        if not (self._are_epochs_valid(epoch_lst)):
            sys.exit()

        # Load net into GPU
        self.device = self._get_cuda_device(0)
        self.net.to(self.device)

        # Loop over each epoch under consideration
        for epoch in epoch_lst:
            # Load checkpoint weights to GPU
            epoch_path = f"{self.workdir}/epoch_{epoch}.pth"
            checkpoint = torch.load(epoch_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])

            # Evaluate
            self.net.eval()
            valloss, valaccu, valtt = self._get_val_loss_and_accuracy()

            # Training accuracy and loss for current epoch from training log
            trnaccu = self._trn_df[self._trn_df['epoch'] ==
                                   epoch]['acc'].item()
            trnloss = self._trn_df[self._trn_df['epoch'] ==
                                   epoch]['loss'].item()

            # Write validation log file
            valstr = (f'{{"mode": "val", "epoch":{epoch}, "acc":{valaccu}, '
                      f'"trnacc":{trnaccu}, "loss":{valloss} }}')
            log.write(f"\n{valstr}")
            log.flush()
            os.fsync(log.fileno())

        now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        log.write(f'\n{{"mode": "info",  "end_time":"{now}" }}')

    def _get_val_loss_and_accuracy(self):
        """ Returns validation accuracy and loss
        """
        with torch.no_grad():
            valloss_lst = []
            valpred_lst = []
            valgt_lst = []
            valtt_lst = []
            for idx, data in enumerate(Bar(self.vloader)):
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
            valaccu = round(accuracy_score(valgt_lst, valpred_lst), 2)

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

    def _are_epochs_valid(self, epoch_lst):
        """ Returns True if we can find all the eopocsh in
        the epoch list
        """
        epoch_path_lst = [f"{self.workdir}/epoch_{x}.pth" for x in epoch_lst]
        for epoch_path in epoch_path_lst:
            if not os.path.isfile(epoch_path):
                return False

        return True

    def _load_trn_json(self):
        """ Load training log file as df
        """
        # Loading training json file
        trn_log_path = f"{self.workdir}/trn_log.json"
        if not os.path.isfile(trn_log_path):
            raise Exception(f"{trn_log_path} does not exist")
        with open(trn_log_path, "r") as f:
            lines = f.readlines()

        # Creating dataframes for training and validation
        trn_dict_lst = []
        for cur_line in lines:
            cur_line_dict = json.loads(cur_line.rstrip())
            if cur_line_dict['mode'] == "train":
                trn_dict_lst += [cur_line_dict]

        # Creating dataframes from list of dictionaries
        trn_df = pd.DataFrame(trn_dict_lst)
        return trn_df
