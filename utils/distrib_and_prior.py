import torch
from utils.optimizer import AdamNormGrad
import os
from scipy.stats import norm, multivariate_normal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import datetime
import numpy as np

def get_prior_(model, args):
   
    with torch.no_grad():
        if args.prior == 'standard':
            dim = args.z1_size if args.model_name == 'vae' else args.z2_size
            means = torch.zeros(dim).unsqueeze(0)
            covs = torch.eye(dim).unsqueeze(0)
            w = torch.Tensor([1])
            idx_valid_mixture = [0]
        else:
            if not args.dpa_training:
                X = model.means(model.idle_input) # C x D
            else:
                X = model.u # C x D

            if args.model_name == 'vae':
                means, logvars = model.q_z(X)  # (C x M), (C x M)
            else:
                if args.model_name=='convhvae_2level' or args.model_name=='pixelhvae_2level':
                    X = X.view(-1, args.input_size[0], args.input_size[1], args.input_size[2])
                means, logvars = model.q_z2(X)
            sds = torch.exp(logvars)
            covs = torch.zeros(sds.size(0), sds.size(1), sds.size(1))
            covs = torch.diag_embed(sds)
            print(means.shape, covs.shape)
            if args.dpa_training==True:
                w = model.w.flatten().cpu().numpy()
            else:
                C = args.number_components
                w = torch.Tensor([1]).expand(C) / C
            idx_valid_mixture =np.squeeze(np.argwhere(w >= 1e-4))
            if idx_valid_mixture.shape == ():
                idx_valid_mixture = [idx_valid_mixture.item()]
            else:
                idx_valid_mixture = idx_valid_mixture.tolist()
            if not args.dpa_training:
                means = means.cpu()
                covs = covs.cpu()
                w = w.cpu()

    return means, covs, w, idx_valid_mixture

def get_aggregated_posterior_(model, args, train_loader):
    if args.cuda:
        device = 'cuda'
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            if args.model_name == 'vae':
                x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = model(data)
                if batch_idx == 0:
                    means = z_q_mean
                    sds = z_q_logvar
                    latents = z_q
                else:
                    means = torch.cat((means,z_q_mean),0)
                    sds = torch.cat((sds, z_q_logvar), 0)
                    latents = torch.cat((latents, z_q), 0)
            else:
                if args.model_name=='convhvae_2level' or args.model_name=='pixelhvae_2level':
                    data = data.view(-1, args.input_size[0], args.input_size[1], args.input_size[2])

                x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = model(data)
                if batch_idx == 0:
                    means = z2_q_mean
                    sds = z2_q_logvar
                    latents = z2_q
                else:
                    means = torch.cat((means,z2_q_mean),0)
                    sds = torch.cat((sds, z2_q_logvar), 0)
                    latents = torch.cat((latents, z2_q), 0)

    sds = torch.exp(sds)
    covs = torch.zeros(sds.size(0), sds.size(1), sds.size(1))
    covs = torch.diag_embed(sds)

    C = means.size(0)
    w = torch.Tensor([1]).expand(C) / C
    idx_valid_mixture = np.squeeze(np.argwhere(w >= -1))
    if idx_valid_mixture.shape == ():
        idx_valid_mixture = [idx_valid_mixture.item()]
    else:
        idx_valid_mixture = idx_valid_mixture.tolist()
    logprobs = torch.distributions.MultivariateNormal(means, covs).log_prob(latents)
    
    return means.cpu(), covs.cpu(), w.cpu(), idx_valid_mixture, latents.cpu(), logprobs

def plot_distribution(dir, args, means, covs, w, samples=None, idx_valid_mixture=None, axis_scale=10, title="", style='density', mode='prior'):
    
    if not idx_valid_mixture:
        idx_valid_mixture = np.squeeze(np.argwhere(w >= 1e-4))
        if idx_valid_mixture.shape == ():
            idx_valid_mixture = [idx_valid_mixture.item()]
        else:
            idx_valid_mixture = idx_valid_mixture.tolist() 
    idx_valid_mixture = idx_valid_mixture[:min(args.NAP, len(idx_valid_mixture))]
    fig, axs = plt.subplots(1, 1, figsize=(6, 6), edgecolor='k')
    if style=='density':
        x, y = np.mgrid[-axis_scale:axis_scale:.05, -axis_scale:axis_scale:.05]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        ticks = np.arange(0, axis_scale*20*2, 40)
        labels = tuple(np.arange(-axis_scale, axis_scale, 2))
        for i in idx_valid_mixture:
            rv = multivariate_normal(means[i].cpu(), covs[i].cpu())
            if i == idx_valid_mixture[0]:
                gm_pdf = rv.pdf(pos)
            else:
                gm_pdf = gm_pdf + rv.pdf(pos)
        gm_pdf = gm_pdf / len(idx_valid_mixture) + 1e-8


        im = axs.imshow(np.log(gm_pdf), cmap='viridis', vmin=-12, vmax=0)
        axs.set_title(f"{title}") 
        axs.set_xticks(ticks)
        axs.set_xticklabels(labels)
        axs.set_yticks(ticks)
        axs.set_yticklabels(labels)
        fig.colorbar(im)
    elif style=='circle':
        axs.scatter(samples[:, 0], samples[:, 1], s=1, c='b')
        for i in idx_valid_mixture:
            draw_ellipse(means[i], covs[i], weight=w[i])
        axs.set_xlim([-axis_scale, axis_scale])
        axs.set_ylim([-axis_scale, axis_scale])
        axs.set(aspect='equal')
        axs.set_title(f"{title}")
    if mode=='prior':
        plt.savefig(f'{dir}prior_density') if style=='density' else plt.savefig(f'{dir}prior_circles')
    elif mode=='aggregated':
        plt.savefig(f'{dir}aggregated_posterior_density') if style=='density' else plt.savefig(f'{dir}aggregated_posterior_circles')
    
def draw_ellipse(position, covariance, weight, ax=None, color='r'):
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    nsig = 2
    ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                         angle=angle, color=color, fill=False, lw=weight * 10))

