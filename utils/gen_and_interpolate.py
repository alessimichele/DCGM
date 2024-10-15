import torch
import os
from torchvision.utils import save_image
from utils.visual_evaluation import plot_images
import random
def generate_from_ps(args, model):

    with torch.no_grad():
        if not args.dpa_training:
            u = model.means(model.idle_input)
        else:
            u = model.u

        # Start with the first pseudoinput
        generations = [u[0].unsqueeze(0)]  # Wrap in a list to prepare for concatenation

        if args.model_name == 'vae':
            # For VAE model
            _, _, z, _, _ = model(u[0].unsqueeze(0))  # Get latent representation z
            for i in range(24):
                x_mean, _ = model.p_x(z)  # Generate image from latent code
                generations.append(x_mean)  # Append each generated image to the list

        else:
            # For other models
            x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = model(u[0].unsqueeze(0))
            z1_sample_mean, z1_sample_logvar = model.p_z1(z2_q)  # Get latent variables
            z1_sample_rand = model.reparameterize(z1_sample_mean, z1_sample_logvar)
            
            for i in range(24):
                if args.model_name == 'pixelhvae_2level':
                    x_mean, _ = model.p_x(u[0].unsqueeze(0).view(-1, args.input_size[0], args.input_size[1], args.input_size[2]), z1_sample_rand, z2_q)
                else:
                    x_mean, _ = model.p_x(z1_sample_rand, z2_q)
                generations.append(x_mean)  # Append each generated image to the list

        # Concatenate all generated images into a single tensor along the 0th dimension
        generations = torch.cat(generations, dim=0)

    return generations


def ps_interpolation(args, model):
    
    with torch.no_grad():
        if not args.dpa_training:
            u = model.means(model.idle_input)
        else:
            u = model.u

        if u.shape[0] > 1:
            num_pseudoinputs = u.shape[0]  # Number of pseudoinputs
            indices = random.sample(range(num_pseudoinputs), 2)  # Two random distinct indices

            first_ps = u[indices[0]]
            second_ps = u[indices[1]]
        else:
            first_ps = u[0]
            second_ps = u[0]

        z_values = torch.linspace(0, 1, steps=10)  # Create 11 equally spaced values from 0 to 1
        inter = torch.stack([z * first_ps + (1 - z) * second_ps for z in z_values], dim=0)

        if args.model_name == 'vae':
            _, _, z, _, _ = model(inter)
            x_mean, _ = model.p_x(z)
            interpolations = x_mean

        else:
            x_mean, x_logvar, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = model(inter)
            z1_sample_mean, z1_sample_logvar = model.p_z1(z2_q)
            z1_sample_rand = model.reparameterize(z1_sample_mean, z1_sample_logvar)

            if args.model_name == 'pixelhvae_2level':
                x_mean, _ = model.p_x(inter.view(-1, args.input_size[0], args.input_size[1], args.input_size[2]), z1_sample_rand, z2_q)
            else:
                x_mean, _ = model.p_x(z1_sample_rand, z2_q)

            interpolations = x_mean

    return interpolations
