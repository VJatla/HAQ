import sys
import pdb
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class DyadicCNN3DV2(nn.Module):

    MAX_DYADS = 4
    """ Maximum dyads possible. This value depends on size of input and maxpooling layer.
    For 224x224 we can have a maximum of 4 dyads """

    
    def __init__(self, num_dyads, in_shape, nkernels=[4, 8, 16, 32],
                 seed=random.randint(101, 201)):
        """ Dyadic 3DCNNs for recognizing activities in AOLME dataset.
        Maximum number of dyads is 6.


        Parameters
        ----------
        num_dyads: int
            Number of dyads in the model. There can be a maximum of
            6 dyads.
        in_shape: int[lst]
            Input shape **[<channels>, <frames>, <Height>, <Width>]**.
            For example, `[3, 90, 224, 224]`.
        nkernels: int[lst], optional
            Number of kernels to use in each dyad. The default is
            [4, 8, 16, 32, 64, 128].
        seed: int, optional
            sets random seed for gpu and cpu
        Note
        ----
        1. One **dyad** is,
        ```
        [3DCNN] -> [BN] -> [ReLU] -> [3D Max pool]
        ```
        2. Kernel size is hard coded to 3.
        """
        # Set random seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # --> not needed. Just in case.

        # Are number of dyads valid?
        if num_dyads > self.MAX_DYADS or num_dyads <= 0:
            raise Exception(f"ERROR: Number of dyads should be"
                            f" [1, {self.MAX_DYADS}]")

        # Initialize pytroch class
        super(DyadicCNN3DV2, self).__init__()

        # Build model
        self.model = nn.ModuleDict(
            self._build_model_dict(num_dyads, nkernels, in_shape))

    def forward(self, x):
        """ Forward pass.
        """
        for lname in self.model:
            layer = self.model[lname]
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
                if layer.out_features > 1:
                    x = layer(x)  # Dense layer
                else:
                    x = torch.sigmoid(layer(x))  # Sigmoid layer
            else:
                raise Exception(f"{lname} is not supported")
        return x


    def _build_model_dict(self, nd, nk, ishape):
        """ Build dyadic networks

        Parameters
        ----------
        nd: int
            Number of dyads in the model. There can be a maximum of
            6 dyads.
        nk: int[lst], optional
            Number of kernels to use in each dyad. The default is
            [4, 8, 16, 32, 64, 128].
        ishape: int[lst]
            Input shape **[<channels>, <frames>, <Height>, <Width>]**.
            It can build models for following input shapes,
                1. [3, 90, 224, 224]
                2. [3, 60, 224, 224]
                3. [3, 30, 224, 224]
        """
        ic = ishape[0]
        del ishape[0]
        model_dict = dict()

        # Dyad loop
        for didx in range(0, nd):
            
            if didx == 0:
                ic = ic
                oc = nk[didx]
            else:
                ic = nk[didx - 1]
                oc = nk[didx]

            # CNN3D
            model_dict[f'Conv_{didx}_0'] = nn.Conv3d(in_channels=ic,
                                                   out_channels=oc,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   padding_mode="zeros")
            ic = oc
            oc = oc
            
            # BN
            model_dict[f'BN_{didx}'] = nn.BatchNorm3d(ic)

            # Maxpooling
            if didx == 0:
                
                # For first layer,
                #     td = 90 => MaxPool3D (3,3,3)
                #     td = 60 => MaxPool3D (2,3,3)
                #     td = 30 => MaxPool3D (1,3,3)
                # Temporal dimension
                td = ishape[0]
                
                if td == 90:
                    model_dict[f'MaxPool3D_{didx}'] = nn.MaxPool3d((3,3,3))
                    ishape = [math.floor(x / 3) for x in ishape]
                    
                elif td == 60:
                    model_dict[f'MaxPool3D_{didx}'] = nn.MaxPool3d((2, 3, 3))
                    ishape = [
                        math.floor(ishape[0]/2), math.floor(ishape[1]/3), math.floor(ishape[2]/3)
                    ]
                elif td == 30:
                    model_dict[f'MaxPool3D_{didx}'] = nn.MaxPool3d((1,3,3))
                    ishape = [
                        math.floor(ishape[0]), math.floor(ishape[1]/3), math.floor(ishape[2]/3)
                    ]
                else:
                    sys.exit(f"Input temporal dimension not supported, {td}")

            else:
                
                # Other layers
                model_dict[f'MaxPool3D_{didx}'] = nn.MaxPool3d(3, 3)
                ishape = [math.floor(x / 3) for x in ishape]
                
                
            # For the rest of the layers
            if 0 in ishape:
                raise Exception(f"ERROR: Shape = {ishape} after {didx} dyads")



        # Network proposed in proposal
        # After dyads flatten and give input to dense layer
        model_dict['Flatten'] = nn.Flatten()
        model_dict['Dropout-Flatten'] = nn.Dropout(p=0.5)  # 50% dropout
        ic = np.prod(ishape) * oc
        model_dict['Dense'] = nn.Linear(ic, 1)

        return model_dict
