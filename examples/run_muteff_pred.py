import numpy as np
import pandas as pd
import time
import sys
sys.path.insert(0, "../DeepSequence/")
import model
import helper
import train
import argparse

parser = argparse.ArgumentParser(description="Calculate DeepSequence mutation effect predictions.")
parser.add_argument("--alignment", type=str, required=True, default="./datasets/alignments/BLAT_ECOLX_1_b0.5.a2m",
                    help="Alignment file")
parser.add_argument("--restore", type=str, required=True, default="BLAT_ECOLX",
                    help="Parameter file prefix for restoring model")
parser.add_argument("--mutants", type=str, required=True, default="./datasets/mutations/BLAT_ECOLX_Ranganathan2015_BLAT_ECOLX_hmmerbit_plmc_vae_hmm_results.csv",
                    help="Table of mutants to predict, or fasta file of aligned sequences (ending in .fa)")
parser.add_argument("--output", type=str, required=True, default="./calc_muteff/output/BLAT_ECOLX_output.csv",
                    help="Output file location")
parser.add_argument("--colname", type=str, default="mutation_effect_prediction",
                    help="Output mutation effect column name")
parser.add_argument("--samples", type=int, default=500,
                    help="Number of prediction iterations")
parser.add_argument("--alphabet_type", type=str, default="protein", help="Specify alphabet type")
args = parser.parse_args()

data_params = {
    "alignment_file"           :   args.alignment,
    }

model_params = {
    "bs"                :   100,
    "encode_dim_zero"   :   1500,
    "encode_dim_one"    :   1500,
    "decode_dim_zero"   :   100,
    "decode_dim_one"    :   500,
    "n_latent"          :   30,
    "logit_p"           :   0.001,
    "sparsity"          :   "logit",
    "final_decode_nonlin":  "sigmoid",
    "final_pwm_scale"   :   True,
    "n_pat"             :   4,
    "r_seed"            :   12345,
    "conv_pat"          :   True,
    "d_c_size"          :   40
    }

if __name__ == "__main__":
    print(args.output, type(args.output))
    OUT = str(args.output)
    print(OUT, type(OUT))
    data_helper = helper.DataHelper(alignment_file=data_params["alignment_file"],
                                    working_dir='.',
                                    calc_weights=False,alphabet_type=args.alphabet_type)

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
    
    print ("Model built")
    
    vae_model.load_parameters(file_prefix=args.restore)
    
    print ("Parameters loaded\n\n")
    
    if args.mutants.endswith(".fa"):
        data_helper.custom_sequences(
            args.mutants,
            vae_model, 
            N_pred_iterations=args.samples,
            filename_prefix=OUT
            )
    else:
	data_helper.custom_mutant_matrix(
            args.mutants,
            vae_model,
            N_pred_iterations=args.samples,
            filename_prefix=OUT
            )

