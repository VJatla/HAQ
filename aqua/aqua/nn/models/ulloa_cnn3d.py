import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class UlloaCNN3D(nn.Module):
    def __init__(self, ksize, ndyads, nchannels=3, debug_prints=False):
        """ The following models are inspired by,
        
        *"Computer-assisted analysis of echocardiographic videos of the
        heart with deep learning improves clinical prediction of
        all-cause mortality"* -- **Alvaro Ullao-Cerna**

        This class uses `pytorch`. At the time of authorship pytorch is 
        at version `1.7.0` and cuda `10.2`.

        Parameters
        ----------
        ksize: int
            Kernel size. A 3D cube is created with the `kszie` in
            each dimension.
        ndyads: int
            Number of dyads in current model. Please refer *Note 1*.
        nchannels: int, optional
            Number of channels. Defaults to 3.
        debug_prints: bool, optional
            Prints useful debugging information. Defautls to False.


        Note
        ----
        1. One **dyad** is,  
        ```
        [3DCNN] -> [BN] -> [ReLU] -> [3DCNN] -> [BN] -> [ReLU] -> [3D Max pool]
        ```
        2. BN = Batch Normalization
        3. Number of kernels follow [4, 8, 16, 32, ...] sequence

        Todo
        ----
        - The class should be able to build models dynamically with number of
        dyads and size of kernel (<--- I think size of kernel doesn't change the
        architecture).
        """
        super(UlloaCNN3D, self).__init__()

        # debug_prints
        self.debug_prints = debug_prints

        # Building model dictionary
        self.model_dict = nn.ModuleDict(
            self._build_ulloa_model_dict(ksize, ndyads, nchannels)
            )

    def forward(self, x):
        """ Forward pass.
        """
        for lname in self.model_dict:
            layer = self.model_dict[lname]
            if isinstance(layer, nn.Conv3d):
                x = layer(x)
            elif isinstance(layer, nn.BatchNorm3d):
                x = F.relu(layer(x))
            elif isinstance(layer, nn.MaxPool3d):
                x = layer(x)
            elif isinstance(layer, nn.Flatten):
                x = layer(x)
            elif isinstance(layer, nn.Dropout):
                x = layer(x)
            elif isinstance(layer, nn.Linear):
                x = torch.sigmoid(layer(x)) # Dense layer
            else:
                raise Exception(f"{lname} is not supported")
        return x
    
    def _build_ulloa_model_dict(self, ksize, ndyads, nchannels):
        """ Replicates Alvaro Ulloa work
        """
        model_dict = dict()
        # Dyad Loop: Counting starts from 1 to n
        for didx in range(1, ndyads+1):
            if didx == 1:
                ic = nchannels
                oc = int(math.pow(2,didx+1))
            else:
                ic = int(math.pow(2,didx))
                oc = int(math.pow(2,didx+1))
                
            # Convx_1
            if self.debug_prints:
                print(f"CONV3D {ic} --> {oc} channels")
            model_dict[f'Conv{didx}_1'] = nn.Conv3d(
                in_channels=ic,
                out_channels=oc,
                kernel_size=ksize,
                stride=1,
                padding=1,
                padding_mode="zeros")

            # BNx_1
            ic = oc
            oc = oc
            if self.debug_prints:
                print(f"BatchNorm3D {ic} --> {oc} channels")
            model_dict[f'BN{didx}_1'] = nn.BatchNorm3d(ic)

            # Convx_2
            ic = oc
            oc = oc
            if self.debug_prints:
                print(f"CONV3D {ic} --> {oc} channels")
            model_dict[f'Conv{didx}_2'] = nn.Conv3d(
                in_channels=ic,
                out_channels=oc,
                kernel_size=ksize,
                stride=1,
                padding=1,
                padding_mode="zeros")
            
            # BNx_2
            ic = oc
            oc = oc
            if self.debug_prints:
                print(f"BatchNorm3D {ic} --> {oc} channels")
            model_dict[f'BN{didx}_2'] = nn.BatchNorm3d(ic)
            
            # MaxPool3D
            model_dict[f'3DMaxpool{didx}'] = nn.MaxPool3d(3, stride=3)
                
        # After dyads flatten and give input to dense layer
        model_dict['Flatten'] = nn.Flatten()
        model_dict['Dropout'] = nn.Dropout(p=0.5) # 50% dropout
        model_dict['Dense'] = nn.Linear(960, 1)

        return model_dict
