import numpy as np
import time
import sys
sys.path.insert(0, "../DeepSequence/")
import model
import helper
import train
import argparse

parser = argparse.ArgumentParser(description="Train DeepSequence with SVI.")
parser.add_argument("--dataset", type=str, default="BLAT_ECOLX",
                    help="Dataset name for fitting model.")
parser.add_argument("--neff-override", type=float, default=None,
                    help="Override the model Neff.")
parser.add_argument("--theta-override", type=float, default=None,
                    help="Override the model theta.")
parser.add_argument("--n-latent-override", type=int, default=None,
                    help="Override the model n_latent.")
parser.add_argument("--weights_dir", type=str, default="", help="Location of precomputed weights, if possible")
parser.add_argument("--alignments_dir", type=str, help="Location of alignments")
parser.add_argument("--seed", type=int, help="Random seed override (for model ensembling).")
args = parser.parse_args()

args.dataset = args.dataset.split(".a2m")[0]

data_params = {
    "dataset"       :   args.dataset,
    "weights_dir"	:   args.weights_dir,
    }

if args.seed is not None:
    print("Using seed: {}".format(args.seed))

model_params = {
    "bs"                :   100,
    "encode_dim_zero"   :   1500,
    "encode_dim_one"    :   1500,
    "decode_dim_zero"   :   100,
    "decode_dim_one"    :   2000,  # 500 in the repo
    "n_latent"          :   30,
    "logit_p"           :   0.001,
    "sparsity"          :   "logit",
    "final_decode_nonlin":  "sigmoid",
    "final_pwm_scale"   :   True,
    "n_pat"             :   4,
    "r_seed"            :   args.seed if args.seed is not None else 12345,
    "conv_pat"          :   True,
    "d_c_size"          :   40
    }

train_params = {
    "num_updates"       :   300000,
    "save_progress"     :   True,
    "verbose"           :   True,
    "save_parameters"   :   50000,
    }

if args.n_latent_override:
    model_params['n_latent'] = args.n_latent_override

if __name__ == "__main__":
    start_time = time.time()
    data_helper = helper.DataHelper(dataset=data_params["dataset"],
                                    working_dir='.',
                                    calc_weights=False,  # Use precomputed weights
                                    theta=args.theta_override,
				                    weights_dir=data_params["weights_dir"],
                                    alignments_dir=args.alignments_dir,
    )
    print("Data loaded.")
    if args.neff_override:
        data_helper.Neff = args.neff_override

    vae_model   = model.VariationalAutoencoder(data_helper,
        batch_size                     =   model_params["bs"],
        encoder_architecture           =   [model_params["encode_dim_zero"],
                                                model_params["encode_dim_one"]],
        decoder_architecture           =   [model_params["decode_dim_zero"],
                                                model_params["decode_dim_one"]],
        n_latent                       =   model_params["n_latent"],
        logit_p                        =   model_params["logit_p"],
        sparsity                       =   model_params["sparsity"],
        encode_nonlinearity_type       =   "relu",
        decode_nonlinearity_type       =   "relu",
        final_decode_nonlinearity      =   model_params["final_decode_nonlin"],
        final_pwm_scale                =   model_params["final_pwm_scale"],
        conv_decoder_size              =   model_params["d_c_size"],
        convolve_patterns              =   model_params["conv_pat"],
        n_patterns                     =   model_params["n_pat"],
        random_seed                    =   model_params["r_seed"],
        )
    print("Model loaded")

    job_string = helper.gen_job_string(data_params, model_params)
    if args.neff_override:
        job_string += "_neff-" + str(args.neff_override)
    if args.seed is not None:
        job_string += "_seed-" + str(args.seed)

    print ("job string: ", job_string)

    print("Starting training")

    train.train(data_helper, vae_model,
        num_updates             =   train_params["num_updates"],
        save_progress           =   train_params["save_progress"],
        save_parameters         =   train_params["save_parameters"],
        verbose                 =   train_params["verbose"],
        job_string              =   job_string)
    print("Training complete")

    vae_model.save_parameters(file_prefix=job_string)

    print("Done in " + str(time.time() - start_time) + " seconds")
