from __future__ import print_function

import torch
from torch.autograd import Variable

from utils.visual_evaluation import plot_images, plot_latent_interpolation_grid, plot_interpol, plot_latent_space_2d
from utils.distrib_and_prior import get_prior_, get_aggregated_posterior_, plot_distribution
from utils.gen_and_interpolate import generate_from_ps, ps_interpolation
import numpy as np
from utils.perform_experiment import build_latent_space
import time

import os
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def evaluate_vae(args, model, train_loader, data_loader, epoch, dir, mode):
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0
    # set model to evaluation mode
    model.eval()

    # evaluate
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            x = data

            # calculate loss function
            loss, RE, KL = model.calculate_loss(x, average=True)

            evaluate_loss += loss.item()
            evaluate_re += -RE.item()
            evaluate_kl += KL.item()

            # print N digits
            if batch_idx == 1 and mode == 'validation':
                if epoch == 1:
                    if not os.path.exists(dir + 'reconstruction/'):
                        os.makedirs(dir + 'reconstruction/')
                    # VISUALIZATION: plot real images
                    plot_images(args, data.data.cpu().numpy()[0:9], dir + 'reconstruction/', 'real', size_x=3, size_y=3)
                x_mean = model.reconstruct_x(x)
                plot_images(args, x_mean.data.cpu().numpy()[0:9], dir + 'reconstruction/', str(epoch), size_x=3, size_y=3)

    if mode == 'test':
        # load all data
        # grab the test data by iterating over the loader
        # there is no standardized tensor_dataset member across pytorch datasets
        test_data, test_target = [], []
        for data, lbls in data_loader:
            test_data.append(data)
            test_target.append(lbls)
            

        test_data, test_target = [torch.cat(test_data, 0), torch.cat(test_target, 0).squeeze()]

        # grab the train data by iterating over the loader
        # there is no standardized tensor_dataset member across pytorch datasets
        full_data = []
        full_labels = []
        for data, labels in train_loader:
            full_data.append(data)
            full_labels.append(labels)
        full_data = torch.cat(full_data, 0)
        full_labels = torch.cat(full_labels, 0)

        latent_space = build_latent_space(args, model, train_loader)
        full_labels = full_labels.squeeze()
        full_labels = full_labels[:latent_space.shape[0]]
        assert latent_space.shape[0] == len(full_labels)
        latent_space, full_labels = latent_space.cpu().numpy(), full_labels.cpu().numpy()
        if args.model_name == 'vae':
            if args.z1_size == 2:
                plot_latent_space_2d(dir, latent_space, full_labels)
            else:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2)
                latent_space_2d = tsne.fit_transform(latent_space)
                plot_latent_space_2d(dir, latent_space_2d, full_labels)
        else:
            if args.z2_size == 2:
                plot_latent_space_2d(dir, latent_space, full_labels)
            else:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2)
                latent_space_2d = tsne.fit_transform(latent_space)
                plot_latent_space_2d(dir, latent_space_2d, full_labels)

        if args.dataset_name == 'dynamic_mnist' or args.dataset_name=='caltech101silhouettes':
            from sklearn.metrics import silhouette_score
            sil_score = silhouette_score(latent_space, full_labels)
            torch.save(sil_score,dir +  'silhouette.score')
            print(f'Silhouette score: {sil_score}')

        if args.cuda:
            test_data, test_target, full_data = test_data.cuda(), test_target.cuda(), full_data.cuda()

        if args.dynamic_binarization:
            full_data = torch.bernoulli(full_data)

        # print(model.means(model.idle_input))

        # VISUALIZATION: plot real images
        plot_images(args, test_data.data.cpu().numpy()[0:25], dir, 'real', size_x=5, size_y=5)

        # VISUALIZATION: plot reconstructions
        samples = model.reconstruct_x(test_data[0:25])

        plot_images(args, samples.data.cpu().numpy(), dir, 'reconstructions', size_x=5, size_y=5)

        # VISUALIZATION: plot generations
        samples_rand = model.generate_x(25)

        plot_images(args, samples_rand.data.cpu().numpy(), dir, 'generations', size_x=5, size_y=5)

        if args.prior == 'vampprior':
            # VISUALIZE pseudoinputs
            if not args.dpa_training:
                pseudoinputs = model.means(model.idle_input).cpu().data.numpy()
                plot_images(args, pseudoinputs[0:25], dir, 'final_pseudoinputs', size_x=5, size_y=5)
            else:
                pseudoinputs = model.u.cpu().numpy()
                plot_images(args, pseudoinputs[0:min(25, pseudoinputs.shape[0])], dir, f'final_pseudoinputs_C_{str(model.C)}', size_x=5, size_y=5)
                

        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_test = model.calculate_lower_bound(test_data, MB=args.MB)
        t_ll_e = time.time()
        print('Test lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_test, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        elbo_train = model.calculate_lower_bound(full_data, MB=args.MB)
        t_ll_e = time.time()
        print('Train lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_train, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_test = model.calculate_likelihood(test_data, dir, mode='test', S=args.S, MB=args.MB)
        t_ll_e = time.time()
        print('Test log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_test, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_train = 0. #model.calculate_likelihood(full_data, dir, mode='train', S=args.S, MB=args.MB)) #commented because it takes too much time
        t_ll_e = time.time()
        print('Train log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_train, t_ll_e - t_ll_s))

        # PLOT DISTRIBUTIONS
        if args.model_name=='vae':
            if args.z1_size==2:
                means, covs, w, idx_valid_mixture = get_prior_(model, args)
                plot_distribution(dir, args, means, covs, w, idx_valid_mixture=idx_valid_mixture ,style='density', mode='prior', title='Prior density')

                means_agg, covs_agg, w_agg, idx_valid_mixture_agg, _, _ = get_aggregated_posterior_(model, args, train_loader)
                plot_distribution(dir, args, means_agg, covs_agg, w_agg,idx_valid_mixture=idx_valid_mixture_agg, style='density', mode='aggregated', title='Aggregated posterior density')
                plot_latent_interpolation_grid(args, model, dir)
        else:
            if args.z2_size==2:
                means, covs, w, idx_valid_mixture = get_prior_(model, args)
                plot_distribution(dir, args, means, covs, w, idx_valid_mixture=idx_valid_mixture ,style='density', mode='prior', title='Prior density')

                means_agg, covs_agg, w_agg, idx_valid_mixture_agg, _, _ = get_aggregated_posterior_(model, args, train_loader)
                plot_distribution(dir, args, means_agg, covs_agg, w_agg,idx_valid_mixture=idx_valid_mixture_agg, style='density', mode='aggregated', title='Aggregated posterior density')
                plot_latent_interpolation_grid(args, model, dir)

        # PLOT INTERPOALTIONS
        if args.prior != 'standard':
            cond_generations = generate_from_ps(args, model)
            plot_images(args, cond_generations.cpu().numpy(), dir, 'gen_given_ps', size_x=5, size_y=5)

            ps_interpolations = ps_interpolation(args, model)
            plot_interpol(args, ps_interpolations.cpu().numpy(), dir, 'interpol_given_ps', size_x=1, size_y=15)

        

        

    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size
    if mode == 'test':
        return evaluate_loss, evaluate_re, evaluate_kl, log_likelihood_test, log_likelihood_train, elbo_test, elbo_train
    else:
        return evaluate_loss, evaluate_re, evaluate_kl
