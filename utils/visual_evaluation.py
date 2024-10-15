import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
#=======================================================================================================================
def plot_histogram( x, dir, mode ):

    fig = plt.figure()

    # the histogram of the data
    n, bins, patches = plt.hist(x, 100, density=True, facecolor='blue', alpha=0.5)
    plt.xlabel('Log-likelihood value')
    plt.ylabel('Probability')
    plt.grid(True)

    plt.savefig(dir + 'histogram_' + mode + '.png', bbox_inches='tight')
    plt.close(fig)

#=======================================================================================================================
def plot_interpol(args, x_sample, dir, file_name, size_x=3, size_y=3):
    fig, axes = plt.subplots(1, 10, figsize=(20, 5))  # Adjust figsize as needed
    
    for i, sample in enumerate(x_sample): 
        sample = sample.reshape((args.input_size[0], args.input_size[1], args.input_size[2]))
        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)
        if args.input_type == 'binary' or args.input_type == 'gray':
            sample = sample[:, :, 0]
            axes[i].imshow(sample, cmap='gray')
        else:
            axes[i].imshow(sample)
        axes[i].axis('off')  # Turn off axis labels for cleaner output

    plt.tight_layout()  # Adjust spacing between plots
    plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    plt.close(fig)
 
def plot_images(args, x_sample, dir, file_name, size_x=3, size_y=3):

    if size_x==size_y:
        fig = plt.figure(figsize=(size_x, size_y))
    else:
        fig = plt.figure(figsize=(2, 8))
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(x_sample):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample = sample.reshape((args.input_size[0], args.input_size[1], args.input_size[2]))
        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)
        if args.input_type == 'binary' or args.input_type == 'gray':
            sample = sample[:, :, 0]
            plt.imshow(sample, cmap='gray')
        else:
            plt.imshow(sample)
    
    plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    plt.close(fig)

def plot_latent_interpolation_grid(args, model, dir, n=10, scale=3.0, figsize=15):
    import torch
    import matplotlib.pyplot as plt
    
    digit_size_1 = args.input_size[1]
    digit_size_2 = args.input_size[2]

    figure = np.zeros((digit_size_1 * n, digit_size_2 * n))
    grid_x = torch.linspace(-scale, scale, n)
    grid_y = torch.linspace(-scale, scale, n)
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z = torch.tensor([[xi, yi]], dtype=torch.float).to('cuda')
            with torch.no_grad():
                if args.model_name=='vae':
                    x_mean, _ = model.p_x(z)
                else:
                    z1_sample_mean, z1_sample_logvar = model.p_z1(z)
                    z1_sample_rand = model.reparameterize(z1_sample_mean, z1_sample_logvar)
                    if args.model_name == 'pixelhvae_2level':
                        x_mean = model.pixelcnn_generate(z1_sample_rand, z)
                    else:
                        x_mean, _ = model.p_x(z1_sample_rand, z)
            x_mean = x_mean.reshape((args.input_size[0], args.input_size[1], args.input_size[2]))
            x_mean = x_mean.swapaxes(0, 2)
            x_mean = x_mean.swapaxes(0, 1)
            if args.input_type == 'binary' or args.input_type == 'gray':
                x_mean = x_mean[:, :, 0]
            
            figure[i * digit_size_1 : (i + 1) * digit_size_1, j * digit_size_2 : (j + 1) * digit_size_2,] = x_mean.cpu().numpy()

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size_2 // 2
    end_range = n * digit_size_2 + start_range
    pixel_range = np.arange(start_range, end_range, digit_size_2)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
    plt.savefig(dir + 'latent_space_interpolation' + '.png', bbox_inches='tight')
    plt.close()

def plot_latent_space_2d(dir, latent_space, full_labels):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_space[:, 0], latent_space[:, 1], c=full_labels, cmap='viridis', alpha=0.5)
    #plt.colorbar(scatter, label='Labels')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()
    plt.savefig(dir + 'latent_space_projection.png', bbox_inches='tight')
