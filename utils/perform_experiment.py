from __future__ import print_function

import torch

import math

import time


from utils.visual_evaluation import plot_images

import os
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, dir, model_name='vae'):
    from utils.training import train_vae as train
    from utils.evaluation import evaluate_vae as evaluate

    # SAVING
    torch.save(args, dir + args.model_name + '.config')

    # best_model = model
    best_loss = 100000.
    e = 0
    train_loss_history = []
    train_re_history = []
    train_kl_history = []

    val_loss_history = []
    val_re_history = []
    val_kl_history = []

    time_history = []

    for epoch in range(1, args.epochs + 1):
        time_start = time.time()
        model, train_loss_epoch, train_re_epoch, train_kl_epoch = train(epoch, args, train_loader, model,
                                                                             optimizer)

        val_loss_epoch, val_re_epoch, val_kl_epoch = evaluate(args, model, train_loader, val_loader, epoch, dir, mode='validation')
        time_end = time.time()

        time_elapsed = time_end - time_start

        # appending history
        train_loss_history.append(train_loss_epoch), train_re_history.append(train_re_epoch), train_kl_history.append(
            train_kl_epoch)
        val_loss_history.append(val_loss_epoch), val_re_history.append(val_re_epoch), val_kl_history.append(
            val_kl_epoch)
        time_history.append(time_elapsed)

        # printing results
        print('Epoch: {}/{}, Time elapsed: {:.2f}s\n'
              '* Train loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
              'o Val.  loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
              '--> Early stopping: {}/{} (BEST: {:.2f})\n'.format(
            epoch, args.epochs, time_elapsed,
            train_loss_epoch, train_re_epoch, train_kl_epoch,
            val_loss_epoch, val_re_epoch, val_kl_epoch,
            e, args.early_stopping_epochs, best_loss
        ))
        

        # early-stopping
        if val_loss_epoch < best_loss:
            e = 0
            best_loss = val_loss_epoch
            # best_model = model
            print('->model saved<-')
            torch.save(model, dir + args.model_name + '.model')
        else:
            e += 1
            if epoch < args.warmup:
                e = 0
            if e > args.early_stopping_epochs:
                break

        # NaN
        if math.isnan(val_loss_epoch):
            break

    # FINAL EVALUATION
    best_model = torch.load(dir + args.model_name + '.model')
    test_loss, test_re, test_kl, test_log_likelihood, train_log_likelihood, test_elbo, train_elbo = evaluate(args, best_model, train_loader, test_loader, 9999, dir, mode='test')

    print('FINAL EVALUATION ON TEST SET\n'
          'LogL (TEST): {:.2f}\n'
          'LogL (TRAIN): {:.2f}\n'
          'ELBO (TEST): {:.2f}\n'
          'ELBO (TRAIN): {:.2f}\n'
          'Loss: {:.2f}\n'
          'RE: {:.2f}\n'
          'KL: {:.2f}'.format(
        test_log_likelihood,
        train_log_likelihood,
        test_elbo,
        train_elbo,
        test_loss,
        test_re,
        test_kl
    ))

    with open('vae_experiment_log.txt', 'a') as f:
        print('FINAL EVALUATION ON TEST SET\n'
          'LogL (TEST): {:.2f}\n'
          'LogL (TRAIN): {:.2f}\n'
          'ELBO (TEST): {:.2f}\n'
          'ELBO (TRAIN): {:.2f}\n'
          'Loss: {:.2f}\n'
          'RE: {:.2f}\n'
          'KL: {:.2f}'.format(
        test_log_likelihood,
        train_log_likelihood,
        test_elbo,
        train_elbo,
        test_loss,
        test_re,
        test_kl
        ), file=f)

    # SAVING
    torch.save(train_loss_history, dir + args.model_name + '.train_loss')
    torch.save(train_re_history, dir + args.model_name + '.train_re')
    torch.save(train_kl_history, dir + args.model_name + '.train_kl')
    torch.save(val_loss_history, dir + args.model_name + '.val_loss')
    torch.save(val_re_history, dir + args.model_name + '.val_re')
    torch.save(val_kl_history, dir + args.model_name + '.val_kl')
    torch.save(test_log_likelihood, dir + args.model_name + '.test_log_likelihood')
    torch.save(test_loss, dir + args.model_name + '.test_loss')
    torch.save(test_re, dir + args.model_name + '.test_re')
    torch.save(test_kl, dir + args.model_name + '.test_kl')


#####################################################################################################################################################################################################################
#####################################################################################################################################################################################################################
#####################################################################################################################################################################################################################


def build_latent_space(args, model, train_loader):
    nbatches = len(train_loader)
    stop_at = int(nbatches * args.perc_latent_space)
    DATA = []

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

    # Concatenate all collected tensors
    if DATA:
        DATA = torch.cat(DATA, dim=0)
    DATA.requires_grad = False
    print('lat_shape: ', DATA.shape)
    return DATA

def experiment_dpavae(args, train_loader, val_loader, test_loader, model, optimizer, dir, model_name='vae'):
    from utils.training import train_vae as train
    from utils.evaluation import evaluate_vae as evaluate
    from  utils.dpa import  to_numpy_float64, get_dpa, get_Zkiller, heuristic
    import torch.nn.functional as F
    import numpy as np
    # SAVING
    torch.save(args, dir + args.model_name + '.config')

    # best_model = model
    best_loss = 100000.
    e = 0
    train_loss_history = []
    train_re_history = []
    train_kl_history = []

    val_loss_history = []
    val_re_history = []
    val_kl_history = []

    time_history = []
    Z_history = []
    C_history = []
    SCORE_history = []

    # --------------------------------------------------------------------------------------------------------------------------------
    Zpar = 0.0
    SCORE = 0.0
    # --------------------------------------------------------------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        time_start = time.time()
        model, train_loss_epoch, train_re_epoch, train_kl_epoch = train(epoch, args, train_loader, model,
                                                                             optimizer)

        val_loss_epoch, val_re_epoch, val_kl_epoch = evaluate(args, model, train_loader, val_loader, epoch, dir, mode='validation')
        time_end = time.time()

        time_elapsed = time_end - time_start

        # appending history
        train_loss_history.append(train_loss_epoch), train_re_history.append(train_re_epoch), train_kl_history.append(
            train_kl_epoch)
        val_loss_history.append(val_loss_epoch), val_re_history.append(val_re_epoch), val_kl_history.append(
            val_kl_epoch)
        time_history.append(time_elapsed)

        # printing results
        print('Epoch: {}/{}, Time elapsed: {:.2f}s\n'
              '* Train loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
              'o Val.  loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
              '--> Early stopping: {}/{} (BEST: {:.2f})\n'.format(
            epoch, args.epochs, time_elapsed,
            train_loss_epoch, train_re_epoch, train_kl_epoch,
            val_loss_epoch, val_re_epoch, val_kl_epoch,
            e, args.early_stopping_epochs, best_loss
        ))


        # --------------------------------------------------------------------------------------------------------------------------------
        # DPA update!
        # --------------------------------------------------------------------------------------------------------------------------------

        # save the pseudoinputs at the beginning
        if epoch ==1:
            plot_images(args, model.u.cpu().numpy(), dir, 'start_pseudoinputs', size_x=int(np.sqrt(model.u.size(0)))+1 , size_y = int(np.sqrt(model.u.size(0)))+1)

        if not os.path.exists(dir + 'pseudoinputs/'):
            os.makedirs(dir + 'pseudoinputs/')
        
        if epoch%args.dpa_step==0 and epoch!=1: 
            with torch.no_grad():
                print('Updating psuedoinputs running DPA in the latent space...')
                Zspace = build_latent_space(args, model, train_loader)
                Zspace = Zspace.cpu()
                dpa = get_dpa(data=to_numpy_float64(Zspace), Zpar=Zpar, verbose=False)
                print('Latent intrinsic dimension:', dpa.intrinsic_dim)

                model.C = len(dpa.cluster_centers) # update C
                print('New number of clusters:', model.C)

                # get the clusters centers in the latent space as candidate latent pseudoinputs
                latent_pseudo_inputs = Zspace[dpa.cluster_centers]
                assert isinstance(latent_pseudo_inputs, torch.Tensor)
                assert latent_pseudo_inputs.shape[0] == model.C
                if args.model_name=='vae':
                    assert latent_pseudo_inputs.shape[1] == args.z1_size
                else:
                    if args.latent_level == 2:
                        assert latent_pseudo_inputs.shape[1] == args.z2_size
                    elif args.latent_level == 1:
                        assert latent_pseudo_inputs.shape[1] == args.z1_size
                # map the latent pseudoinputs to the data space, to get the new pseudoinputs
                if args.cuda:
                    latent_pseudo_inputs = latent_pseudo_inputs.cuda()

                if args.model_name=='vae':
                    model.u, _ = model.p_x(latent_pseudo_inputs) 
                else:
                    if args.latent_level == 2:
                        z2_mean, z2_logvar = model.p_z1(latent_pseudo_inputs)
                        z1_rep = model.reparameterize(z2_mean, z2_logvar)
                        if args.model_name=='pixelhvae_2level':
                            model.u = model.pixelcnn_generate( z1_rep, latent_pseudo_inputs)
                            model.u = model.u.view(-1, np.prod(args.input_size))
                            assert model.u.shape[1] == np.prod(args.input_size)
                        else:
                            model.u, _ = model.p_x(z1_rep, latent_pseudo_inputs)
                            assert model.u.shape[1] == np.prod(args.input_size)
                    elif args.latent_level == 1:
                        z2_sample_rand = torch.randn(latent_pseudo_inputs.shape[0], args.z2_size, requires_grad=True)
                        if args.cuda:
                            z2_sample_rand = z2_sample_rand.cuda()
                        
                        if args.model_name=='pixelhvae_2level':
                            model.u = model.pixelcnn_generate( latent_pseudo_inputs, z2_sample_rand)
                            model.u = model.u.view(-1, np.prod(args.input_size))
                            assert model.u.shape[1] == np.prod(args.input_size)
                        else:
                            model.u, _ = model.p_x(latent_pseudo_inputs, z2_sample_rand)
                            assert model.u.shape[1] == np.prod(args.input_size)


                assert model.u.shape[0] == model.C

                # update the mixing coefficients
                w = get_Zkiller(dpa)
                w = torch.tensor(w) # C
                assert w.size(0) == model.C
                w = F.softmax(w, dim=0) # C
                w = w.view(-1,)  # C,  
                model.w = w


                print('Current Zpar:', Zpar)
                print('Computing heuristic score...')
                time0 = time.time()
                assert isinstance(Zspace, torch.Tensor)
                score = heuristic(data=Zspace, Z=Zpar, chunk_size=5)
                print('Time to compute heuristic score:', time.time()-time0)
                print('Heuristic score:', score)

                if score >= SCORE:
                    delta = max(0.05, 1/((epoch-args.warmup)+0.5))
                    Zpar += np.random.normal(0.2, 0.01) # delta
                else:
                    #Zpar = Zpar - Zpar/4 if Zpar > 0.0 else 0.0
                    delta = max(0.05, 1/((epoch-args.warmup)+2+0.5)-0.05)
                    Zpar -= np.random.normal(0.2, 0.01) # delta
                SCORE = score
                print('New Zpar:', Zpar, end='\n')

                Z_history.append(Zpar)
                C_history.append(model.C)
                SCORE_history.append(score)

                if args.Zfixed is not None:
                    Zpar = args.Zfixed
                
                
                plot_images(args, model.u.cpu().numpy(), dir, f'pseudoinputs/ps_{epoch}_{model.C}', size_x=int(np.sqrt(model.u.size(0)))+1 , size_y = int(np.sqrt(model.u.size(0)))+1)

                if args.cuda:
                    model.u = model.u.cuda()
                    model.w = model.w.cuda() # check if float64 is ok with cuda

                del Zspace, latent_pseudo_inputs, dpa, score
                torch.cuda.empty_cache()


        # early-stopping
        if val_loss_epoch < best_loss:
            e = 0
            best_loss = val_loss_epoch
            # best_model = model
            print('->model saved<-')
            torch.save(model, dir + args.model_name + '.model')
        else:
            e += 1
            if epoch < args.warmup:
                e = 0
            if e > args.early_stopping_epochs:
                break

        # NaN
        if math.isnan(val_loss_epoch):
            break


    # --------------------------------------------------------------------------------------------------------------------------------
    # FINAL EVALUATION

    torch.save(train_loss_history, dir + args.model_name + '.train_loss')
    torch.save(train_re_history, dir + args.model_name + '.train_re')
    torch.save(train_kl_history, dir + args.model_name + '.train_kl')
    torch.save(val_loss_history, dir + args.model_name + '.val_loss')
    torch.save(val_re_history, dir + args.model_name + '.val_re')
    torch.save(val_kl_history, dir + args.model_name + '.val_kl')
    torch.save(Z_history, dir + args.model_name + '.Z_history')
    torch.save(C_history, dir + args.model_name + '.C_history')
    torch.save(SCORE_history, dir + args.model_name + '.SCORE_history')

    best_model = torch.load(dir + args.model_name + '.model')
    test_loss, test_re, test_kl, test_log_likelihood, train_log_likelihood, test_elbo, train_elbo = evaluate(args, best_model, train_loader, test_loader, 9999, dir, mode='test')

    print('FINAL EVALUATION ON TEST SET\n'
          'LogL (TEST): {:.2f}\n'
          'LogL (TRAIN): {:.2f}\n'
          'ELBO (TEST): {:.2f}\n'
          'ELBO (TRAIN): {:.2f}\n'
          'Loss: {:.2f}\n'
          'RE: {:.2f}\n'
          'KL: {:.2f}'.format(
        test_log_likelihood,
        train_log_likelihood,
        test_elbo,
        train_elbo,
        test_loss,
        test_re,
        test_kl
    ))

    with open('vae_experiment_log.txt', 'a') as f:
        print('FINAL EVALUATION ON TEST SET\n'
          'LogL (TEST): {:.2f}\n'
          'LogL (TRAIN): {:.2f}\n'
          'ELBO (TEST): {:.2f}\n'
          'ELBO (TRAIN): {:.2f}\n'
          'Loss: {:.2f}\n'
          'RE: {:.2f}\n'
          'KL: {:.2f}'.format(
        test_log_likelihood,
        train_log_likelihood,
        test_elbo,
        train_elbo,
        test_loss,
        test_re,
        test_kl
        ), file=f)

    # SAVING
    torch.save(test_log_likelihood, dir + args.model_name + '.test_log_likelihood')
    torch.save(test_loss, dir + args.model_name + '.test_loss')
    torch.save(test_re, dir + args.model_name + '.test_re')
    torch.save(test_kl, dir + args.model_name + '.test_kl')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
