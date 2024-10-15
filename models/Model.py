from __future__ import print_function

import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from utils.nn import normal_init, NonLinear

from utils.dpa import from_loader_to_tensor, to_numpy_float64, get_dpa, get_Zkiller, heuristic
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

    # AUXILIARY METHODS
    def add_pseudoinputs(self):
        if not self.args.dpa_training:
        
            nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)

            self.means = NonLinear(self.args.number_components, np.prod(self.args.input_size), bias=False, activation=nonlinearity)

            # init pseudo-inputs
            if self.args.use_training_data_init:
                self.means.linear.weight.data = self.args.pseudoinputs_mean
            else:
                normal_init(self.means.linear, self.args.pseudoinputs_mean, self.args.pseudoinputs_std)

            # create an idle input for calling pseudo-inputs
            self.idle_input = torch.eye(self.args.number_components, self.args.number_components, requires_grad=False)
            if self.args.cuda:
                self.idle_input = self.idle_input.cuda()
        else:
            # C: number of pseudo-inputs
            # D: input size
            # Zpar: parameter for DPA
            data = from_loader_to_tensor(self.args.train_loader)
            if self.args.perc_latent_space < 1.0:
                data = data[:min(10000, int(self.args.perc_latent_space*data.shape[0]))]
                print(f'Using {data.shape[0]} data for building the latent space.')
            assert data.shape[1] == np.prod(self.args.input_size), f"Data array has {data.shape[1]} features, but expected {np.prod(self.args.input_size)} features."

            print('Initial Zpar: ', self.args.Zparinit)
            dpa = get_dpa(data, Zpar=self.args.Zparinit, verbose=False)
            self.ID = dpa.intrinsic_dim
            self.C = int(len(dpa.cluster_centers)) # C (number of peaks taken as pseudo-inputs)
            print('Number of pseudo-inputs (C): ', self.C)

            # pseudoinputs
            self.u = data[dpa.cluster_centers] # C x D
            self.u = self.u.float()

            assert self.u.size(0) == self.C
            assert self.u.size(1) == np.prod(self.args.input_size)

            self.u.requires_grad = False

            # mixing weights
            w = get_Zkiller(dpa) # C
            assert len(w) == self.C
            w = torch.tensor(w)
            w = F.softmax(w, dim=0) #print(torch.sum(w, dim=0))
            self.w = w.view(-1,) # C, 
            self.w.requires_grad = False

            # load pseudo-inputs and mixing coefficients to the correct device
            if self.args.cuda:
                self.u = self.u.cuda()
                self.w = self.w.cuda()
            
            del self.args.train_loader # not useful anymore


    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        if self.args.cuda:
            eps = torch.randn_like(std).cuda()
        else:
            eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def calculate_loss(self):
        return 0.

    def calculate_likelihood(self):
        return 0.

    def calculate_lower_bound(self):
        return 0.

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        return 0.

#=======================================================================================================================
