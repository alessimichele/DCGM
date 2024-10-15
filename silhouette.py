import torch
import numpy as np
from torch.utils import data as data_utils
import os
import matplotlib.pyplot as plt


def load_dynamic_mnist(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = True

    # start processing
    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader( datasets.MNIST('../datasets', train=True, download=True,
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor()
                                                               ])),
                                                batch_size=args.batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader( datasets.MNIST('../datasets', train=False,
                                                              transform=transforms.Compose([transforms.ToTensor()
                                                                        ])),
                                               batch_size=args.batch_size, shuffle=False)

    # preparing data
    x_train = train_loader.dataset.data.float().numpy() / 255.
    x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2] ) )

    y_train = np.array( train_loader.dataset.targets.float().numpy(), dtype=int)

    x_test = test_loader.dataset.data.float().numpy() / 255.
    x_test = np.reshape( x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2] ) )

    y_test = np.array( test_loader.dataset.targets.float().numpy(), dtype=int)

    # validation set
    x_val = x_train[50000:60000]
    y_val = np.array(y_train[50000:60000], dtype=int)
    x_train = x_train[0:50000]
    y_train = np.array(y_train[0:50000], dtype=int)

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=False, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # setting pseudo-inputs inits
    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
        init = x_train[0:args.number_components].T
        args.pseudoinputs_mean = torch.from_numpy( init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components) ).float()
    else:
        args.pseudoinputs_mean = 0.05
        args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args


def load_dataset(args, **kwargs):

    if args.dataset_name == 'dynamic_mnist':
        train_loader, val_loader, test_loader, args = load_dynamic_mnist(args, **kwargs)
    elif args.dataset_name == 'omniglot':
        from utils.load_data import load_omniglot
        train_loader, val_loader, test_loader, args = load_omniglot(args, **kwargs)
    elif args.dataset_name == 'caltech101silhouettes':
        from utils.load_data import load_caltech101silhouettes
        train_loader, val_loader, test_loader, args = load_caltech101silhouettes(args, **kwargs)
    elif args.dataset_name == 'histopathologyGray':
        from utils.load_data import load_histopathologyGray
        train_loader, val_loader, test_loader, args = load_histopathologyGray(args, **kwargs)
    elif args.dataset_name == 'freyfaces':
        from utils.load_data import load_freyfaces
        train_loader, val_loader, test_loader, args = load_freyfaces(args, **kwargs)
    elif args.dataset_name == 'cifar10':
        from utils.load_data import load_cifar10
        train_loader, val_loader, test_loader, args = load_cifar10(args, **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')

    return train_loader, val_loader, test_loader, args

def build_latent_space(args, model, train_loader):
    nbatches = len(train_loader)
    stop_at = int(nbatches * args.perc_latent_space)
    DATA = []
    TARGET = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= stop_at:
                break  # Stop once we reach stop_at batches

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            elif args.mps:
                data, target = data.to('mps'), target.float().to('mps')

            # dynamic binarization
            if args.dynamic_binarization:
                x = torch.bernoulli(data)
            else:
                x = data
            
            out = model(x)

            if args.model_name == 'vae':
                lat = out[2]
            else:
                if args.latent_level == 2:
                    lat = out[5]
                elif args.latent_level == 1:
                    lat = out[2]

            DATA.append(lat)
            TARGET.append(target)

    # Concatenate all collected tensors
    if DATA:
        DATA = torch.cat(DATA, dim=0)
        TARGET = torch.cat(TARGET, dim=0)
    DATA.requires_grad = False
    TARGET.requires_grad = False
    TARGET = TARGET.view(-1, )
    print('lat_shape: ', DATA.shape)
    print('target_shape: ', TARGET.shape)
    return DATA, TARGET

def process_run(path):
    # check if the path is a directory and valid
    if not os.path.isdir(path):
        raise ValueError(f'Invalid path: {path}')
    
    for file in os.listdir(path):
        if file.endswith('.config'):
            config = file
        elif file.endswith('.model'):
            mod = file

    args = torch.load(f'{path}/{config}')
    print(args)

    if args.model_name == 'vae':
        from models.VAE import VAE
    elif args.model_name == 'hvae_2level':
        from models.HVAE_2level import VAE
    elif args.model_name == 'convhvae_2level':
        from models.convHVAE_2level import VAE
    elif args.model_name == 'pixelhvae_2level':
        from models.PixelHVAE_2level import VAE

    FLAG = False
    if args.dpa_training==True:
        FLAG = True
    if FLAG:
        args.dpa_training = False
    model = VAE(args)
    if FLAG:
        args.dpa_training = True

    with torch.no_grad():
        model = torch.load(f'{path}/{mod}')

        train_loader, val_loader, test_loader, args = load_dataset(args)
        latent_space, targets = build_latent_space(args, model, train_loader)
    latent_space, targets = latent_space.cpu().numpy(), targets.cpu().numpy()
    
    from sklearn.metrics import silhouette_score
    sil_score = silhouette_score(latent_space, targets)
    return sil_score    





#path = 'snapshots/dynamic_mnist/2024_09_23_09_51_02_dynamic_mnist_pixelhvae_2level_vampprior_K_100__wu_50__z1_40_z2_40'
#process_run(path)


def process_data(base_dir):
    i=0
    for directory in os.listdir(base_dir):
        subdir = os.path.join(base_dir, directory)
       
        if not 'dynamic_mnist' in subdir:
            continue
        if not os.path.isdir(subdir):
            continue
        
        for nested_directory in os.listdir(subdir):
            
        
            if 'K_50_' in nested_directory or 'K_100' in nested_directory or 'K_200' in nested_directory:
                continue
            elif 'VAE' in nested_directory:
                continue
            elif '_z1_2_' in nested_directory or '_z2_2_' in nested_directory:
                continue
            
            path = os.path.join(subdir, nested_directory)
            
            
            i+=1
            sil = process_run(path)
            print(f'{path}: {sil}\n')
            
    print(i)
    return


#process_data('./snapshots')
                             

if __name__ == '__main__':
    #process_data('./snapshots') 
    for run in os.listdir('./snapshots/STANDARDVAE/VAE'):
        path = os.path.join('./snapshots/STANDARDVAE/VAE', run)
        if not 'dynamic_mnist' in path:
            continue
        sil = process_run(path)
        print(f'{path}: {sil}\n')


