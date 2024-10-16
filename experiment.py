from __future__ import print_function
import argparse

import torch
import torch.optim as optim

from utils.optimizer import AdamNormGrad

import os

import datetime

from utils.load_data import load_dataset

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #


# Training settings
parser = argparse.ArgumentParser(description='VAE+VampPrior')
# arguments for optimization
parser.add_argument('--batch_size', type=int, default=100, metavar='BStrain',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='BStest',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=2000, metavar='E',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                    help='number of epochs for early stopping')

parser.add_argument('--warmup', type=int, default=50, metavar='WU',
                    help='number of epochs for warm-up')

# cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
# random seed
parser.add_argument('--seed', type=int, default=14, metavar='S',
                    help='random seed (default: 14)')
# model: latent size, input_size, so on
parser.add_argument('--z1_size', type=int, default=40, metavar='M1',
                    help='latent size')
parser.add_argument('--z2_size', type=int, default=40, metavar='M2',
                    help='latent size')
parser.add_argument('--input_size', type=int, default=[1, 28, 28], metavar='D',
                    help='input size')

parser.add_argument('--activation', type=str, default=None, metavar='ACT',
                    help='activation function')

parser.add_argument('--number_components', type=int, default=500, metavar='NC',
                    help='number of pseudo-inputs')
parser.add_argument('--pseudoinputs_mean', type=float, default=-0.05, metavar='PM',
                    help='mean for init pseudo-inputs')
parser.add_argument('--pseudoinputs_std', type=float, default=0.01, metavar='PS',
                    help='std for init pseudo-inputs')

parser.add_argument('--use_training_data_init', action='store_true', default=False,
                    help='initialize pseudo-inputs with randomly chosen training data')

# model: model name, prior
parser.add_argument('--model_name', type=str, default='vae', metavar='MN',
                    help='model name: vae, hvae_2level, convhvae_2level, pixelhvae_2level')

parser.add_argument('--prior', type=str, default='vampprior', metavar='P',
                    help='prior: standard, vampprior')

parser.add_argument('--input_type', type=str, default='binary', metavar='IT',
                    help='type of the input: binary, gray, continuous')

# dpa
parser.add_argument('--dpa_step', type=int, default=4, metavar='dpastep',
                    help='perform dpa update every dpa_step epochs')
parser.add_argument('--dpa_training', action='store_true', default=False,
                    help='use dpa training')
parser.add_argument('--Zparinit', type=float, default=1.0, metavar='Zpar',
                    help='initial parameter for DPA')
parser.add_argument('--Zfixed', type=float, default=None, metavar='Zf',
                    help='If not None (float), fix the parameter for DPA along epochs')
parser.add_argument('--perc_latent_space', type=float, default=1.0, metavar='pls',
        help='Percentage of the training set used to build the latent space to run DPA. Default to 1.0: 100% of the training set')
parser.add_argument('--latent_level', type=int, default=2, metavar='lat_lev',
        help='Specify the latent level (useful only for hierarchical models) where to run DPA. If 2, run DPA inside the deepest latent space -z2-, if 1 run DPA inside the intermediate latent space -z1.')

# experiment
parser.add_argument('--S', type=int, default=100, metavar='SLL',
                    help='number of samples used for approximating log-likelihood')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                    help='size of a mini-batch used for approximating log-likelihood')
parser.add_argument('--NAP', type=int, default=10000, metavar='NAP',
                    help='Number of samples used to plot the density of the aggregated posterior')

# dataset
parser.add_argument('--dataset_name', type=str, default='freyfaces', metavar='DN',
                    help='name of the dataset: static_mnist, dynamic_mnist, omniglot, caltech101silhouettes, histopathologyGray, freyfaces, cifar10')

parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                    help='allow dynamic binarization')

parser.add_argument('--save_dir', type=str, default=None, metavar='savedir',
                    help='where to save the results, inside snapshots/ dir')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run(args, kwargs):
    args.model_signature = str(datetime.datetime.now())[0:19]
    args.model_signature = args.model_signature.replace(" ", "_").replace(":", "_").replace("-", "_")
   
    if not args.dpa_training:
        model_name = args.dataset_name + '_' + args.model_name + '_' + args.prior + '_K_' + str(args.number_components) + '_' + '_wu_' + str(args.warmup) + '_' + '_z1_' + str(args.z1_size) + '_z2_' + str(args.z2_size) 
    else:
        model_name = args.dataset_name + '_dpa_' + args.model_name + '_' + args.prior +  '_wu_' + str(args.warmup) + '_' + '_z1_' + str(args.z1_size) + '_z2_' + str(args.z2_size) + '_perclat_' + str(args.perc_latent_space) + '_dpalevel_' + str(args.latent_level) + '_dpastep_' + str(args.dpa_step)  

    # DIRECTORY FOR SAVING
    if args.save_dir is not None:
        snapshots_path = f'snapshots/{args.save_dir}/'
    else:
        snapshots_path = f'snapshots/{args.dataset_name}/'
    dir = snapshots_path + args.model_signature + '_' + model_name +  '/'

    if not os.path.exists(dir):
        os.makedirs(dir)

    # LOAD DATA=========================================================================================================
    print('load data')

    # loading data
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)
    
    if args.dpa_training:
        args.train_loader = train_loader

    # CREATE MODEL======================================================================================================
    print('create model')
    # importing model
    if args.model_name == 'vae':
        from models.VAE import VAE
    elif args.model_name == 'hvae_2level':
        from models.HVAE_2level import VAE
    elif args.model_name == 'convhvae_2level':
        from models.convHVAE_2level import VAE
    elif args.model_name == 'pixelhvae_2level':
        from models.PixelHVAE_2level import VAE
    else:
        raise Exception('Wrong name of the model!')

    model = VAE(args)
    if args.cuda:
        model.cuda()

    optimizer = AdamNormGrad(model.parameters(), lr=args.lr)

    # ======================================================================================================================
    print(args)
    with open('vae_experiment_log.txt', 'a') as f:
        print(args, file=f)

    # ======================================================================================================================
    print('perform experiment')
    if not args.dpa_training or args.prior=='standard':
        from utils.perform_experiment import experiment_vae
        experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, dir, model_name = args.model_name)
    else:
        from utils.perform_experiment import experiment_dpavae
        print('DPA training')
        experiment_dpavae(args, train_loader, val_loader, test_loader, model, optimizer, dir, model_name = args.model_name)
    # ======================================================================================================================
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    with open('vae_experiment_log.txt', 'a') as f:
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n', file=f)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    run(args, kwargs)

# # # # # # # # # # #
# END EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #
