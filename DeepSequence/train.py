from __future__ import print_function
import numpy as np
import time

def train(data,
    model,
    save_progress=False,
    save_parameters=False,
    num_updates=300000,
    verbose=True,
    job_string="",
    embeddings=False,
    update_offset=0,
    print_neff=True,
    print_iter=100):

    """
    Main function to train DeepSequence

    Parameters
    --------------
    data: Instance of DataHelper class from helper.py
    model: Instance of VariationalAutoencoder or VariationalAutoencoderMLE
            from model.py
    save_progress: save log files of losses during training
    save_parameters: save parameters every k iterations
    num_updates: Number of training iterations (int)
    verbose: Print output during training
    job_string: string by which to save all summary files during training
    embeddings: save latent variables every k iterations (int)
                or "log": save latent variables during training on log scale iterations
                or False (bool)
    update_offset: Offset use for Adam in training
                        Change this to keep training parameters from an old model
    print_neff: Print the Neff of the alignment
    print_iter: print/write out losses every k iterations

    Returns
    ------------
    None
    """

    batch_size = model.batch_size
    batch_order = np.arange(data.x_train.shape[0])

    seq_sample_probs = data.weights / np.sum(data.weights)

    update_num = 0

    # If I am starting a model from a later timestep, fix it here
    if update_offset != 0:
        model.t = update_offset

    LB_list = []
    loss_params_list = []
    KLD_latent_list = []
    reconstruct_list = []

    if save_progress:
        err_filename = data.working_dir+"/logs/"+job_string+"_err.csv"
        OUTPUT = open(err_filename, "w")
        if print_neff:
            OUTPUT.write("Neff:\t"+str(data.Neff)+"\n")
        OUTPUT.close()

    start = time.time()

    # calculate the indices to save the latent variables of all the sequences
    #   in the alignment during training
    #  We found that saving these parameters on the log scale of training calls
    #      is most interpretable
    if embeddings == "log":
        start_embeddings = 10
        log_embedding_interpolants = sorted(list(set(np.floor(np.exp(\
            np.linspace(np.log(start_embeddings),np.log(50000),1250))).tolist())))
        log_embedding_interpolants = [int(val) for val in log_embedding_interpolants]

    # Training
    while (update_num + update_offset) < num_updates:
        update_num += 1

        batch_index = np.random.choice(batch_order, batch_size, \
            p=seq_sample_probs).tolist()

        batch_LB, batch_reconstr_entropy, batch_loss_params, batch_KLD_latent \
            = model.update(data.x_train[batch_index], data.Neff, update_num)

        LB_list.append(batch_LB)
        loss_params_list.append(batch_loss_params)
        KLD_latent_list.append(batch_KLD_latent)
        reconstruct_list.append(batch_reconstr_entropy)

        if save_parameters != False and update_num % save_parameters == 0:
            if verbose:
                print ("Saving Parameters")
            model.save_parameters(job_string+"_epoch-"\
                +str(update_num+update_offset))

        # Make embeddings in roughly log-time
        if embeddings != False:

            if embeddings == "log":
                if update_num + update_offset in log_embedding_interpolants:
                    data.get_embeddings(model, update_num + update_offset, \
                        filename_prefix=job_string)
            else:
                if update_num % embeddings == 0:
                    data.get_embeddings(model, update_num + update_offset, \
                        filename_prefix=job_string)

        if update_num != 0 and update_num % print_iter == 0:

            mean_index = np.arange(update_num-print_iter,update_num)

            LB = np.mean(np.asarray(LB_list)[mean_index])
            KLD_params = np.mean(np.asarray(loss_params_list)[mean_index])
            KLD_latent = np.mean(np.asarray(KLD_latent_list)[mean_index])
            reconstruct = np.mean(np.asarray(reconstruct_list)[mean_index])

            progress_string = "Update {0} finished. LB : {1:.2f},  Params: {2:.2f} , Latent: {3:.2f}, Reconstruct: {4:.2f}, Time: {5:.2f}".format(update_num+update_offset,\
                LB, KLD_params, KLD_latent, reconstruct, time.time() - start)


            start = time.time()

            if verbose:
                print (progress_string)

            if save_progress:
                OUTPUT = open(err_filename, "a")
                OUTPUT.write(progress_string+"\n")
                OUTPUT.close()
