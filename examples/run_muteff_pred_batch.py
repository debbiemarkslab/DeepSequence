# Similar to run_muteff_pred, but using a CSV to specify the MSA and DMS inputs.

import argparse
import os
import time
import sys

import pandas as pd

sys.path.insert(0, "../DeepSequence/")
import model
import helper
import train

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


def create_parser():
    parser = argparse.ArgumentParser(description="Calculate DeepSequence mutation effect predictions.")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Number of prediction iterations")
    parser.add_argument("--alphabet_type", type=str, default="protein", help="Specify alphabet type")

    # New arguments
    parser.add_argument("--dms_input_dir", type=str, required=True)
    parser.add_argument("--dms_output_dir", type=str, required=True)
    parser.add_argument("--msa_path", type=str, required=True)
    parser.add_argument("--dms_index", type=int, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--dms_mapping", type=str, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--msa_use_uniprot", action="store_true")

    # Mirror batch_size default of 2000 in helper.py
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size for ELBO estimation")
    return parser


def main(args):
    # Load the deep mutational scan
    assert os.path.isdir(args.model_checkpoint), "Model checkpoint directory does not exist:" + args.model_checkpoint
    assert args.dms_index is not None, "Must specify a dms index"
    assert os.path.isfile(args.dms_mapping), "Mapping file does not exist:" + args.dms_mapping
    assert os.path.isdir(args.dms_input_dir), "DMS input directory does not exist:" + args.dms_input_dir
    assert os.path.isdir(args.dms_output_dir), "DMS output directory does not exist:" + args.dms_output_dir

    if args.seed is not None:
        print("Using seed:", args.seed)
        model_params['r_seed'] = args.seed

    DMS_phenotype_name, dms_input, dms_output, msa_path, mutant_col, sequence = get_dms_mapping(args)

    # Don't need weights, just using the DataHelper to get the focus_seq_trimmed
    data_helper = helper.DataHelper(alignment_file=msa_path,
                                    working_dir='..',
                                    calc_weights=False,
                                    alphabet_type=args.alphabet_type,
                                    )
    assert sequence != data_helper.focus_seq_trimmed, "Sequence in DMS file does not match sequence in MSA file"

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
    print("Model skeleton built")

    # Could also use Uniprot ID here
    print("Using MSA path as model prefix: " + msa_path)
    dataset_name = msa_path.split('.a2m')[0].split('/')[-1]
    print("Searching for dataset {} in checkpoint dir: {}".format(dataset_name, args.model_checkpoint))
    # TODO not using model_params["r_seed"] here
    vae_model.load_parameters(file_prefix="dataset-" + str(dataset_name), seed=args.seed,
                              override_params_dir=args.model_checkpoint)

    print("Parameters loaded\n\n")

    # Note(Lood): This should probably return the df, and then we can save it outside of the function
    data_helper.custom_mutant_matrix_df(
        input_filename=dms_input,
        model=vae_model,
        mutant_col=mutant_col,
        effect_col=DMS_phenotype_name,
        N_pred_iterations=args.samples,
        minibatch_size=args.batch_size,
        output_filename_prefix=dms_output,
        silent_allowed=True,
        random_seed=args.seed,
    )


def get_dms_mapping(args):
    mapping_protein_seq_DMS = pd.read_csv(args.dms_mapping)
    DMS_id = mapping_protein_seq_DMS["DMS_id"][args.dms_index]
    print("Compute scores for DMS: " + str(DMS_id))
    sequence = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0].upper()
    dms_input = os.path.join(args.dms_input_dir, mapping_protein_seq_DMS["DMS_filename"][
        mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0])
    assert os.path.isfile(dms_input), "DMS input file not found" + dms_input
    print("DMS input file: " + dms_input)
    if "DMS_mutant_column" in mapping_protein_seq_DMS.columns:
        mutant_col = mapping_protein_seq_DMS["DMS_mutant_column"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0]
    else:
        print("DMS_mutant_column not found in mapping file, using mutant")
        mutant_col = "mutant"
    DMS_phenotype_name = \
        mapping_protein_seq_DMS["DMS_phenotype_name"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0]
    print("DMS mutant column: " + mutant_col)
    print("DMS phenotype name: " + DMS_phenotype_name)
    dms_output = os.path.join(args.dms_output_dir, DMS_id)  # Only the prefix, as the file will have _samples etc
    msa_path = os.path.join(args.msa_path,
                            mapping_protein_seq_DMS["MSA_filename"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[
                                0])  # msa_path is expected to be the path to the directory where MSAs are located.

    # Rather use UniProt ID as MSA prefix
    if args.msa_use_uniprot:
        if not os.path.isfile(msa_path):
            print("MSA file not found: " + msa_path)
            print("Looking for close match using UniProt ID")
            uniprot_id = mapping_protein_seq_DMS["UniProt_ID"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0]
            print("UniProt ID: " + uniprot_id)
            found = [file for file in os.listdir(args.msa_path) if
                     file.startswith(uniprot_id) and file.endswith(".a2m")]
            assert len(found) == 1, "Could not find unique MSA file for Uniprot ID {} in {}, found {} files".format(
                uniprot_id, args.msa_path, found)
            msa_path = os.path.join(args.msa_path, found[0])
            print("New MSA file: " + msa_path)

    print("MSA file: " + msa_path)
    assert os.path.isfile(msa_path), "MSA file not found: " + msa_path
    target_seq_start_index = mapping_protein_seq_DMS["start_idx"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0] \
        if "start_idx" in mapping_protein_seq_DMS.columns else 1
    target_seq_end_index = target_seq_start_index + len(sequence)
    # msa_start_index = mapping_protein_seq_DMS["MSA_start"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0] if "MSA_start" in mapping_protein_seq_DMS.columns else 1
    # msa_end_index = mapping_protein_seq_DMS["MSA_end"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0] if "MSA_end" in mapping_protein_seq_DMS.columns else len(args.sequence)
    # if (target_seq_start_index!=msa_start_index) or (target_seq_end_index!=msa_end_index):
    #     args.sequence = args.sequence[msa_start_index-1:msa_end_index]
    #     target_seq_start_index = msa_start_index
    #     target_seq_end_index = msa_end_index
    # df = pd.read_csv(args.dms_input)
    # df,_ = DMS_file_cleanup(df, target_seq=args.sequence, start_idx=target_seq_start_index, end_idx=target_seq_end_index, DMS_mutant_column=mutant_col, DMS_phenotype_name=DMS_phenotype_name)
    # else:
    #     df = pd.read_csv(args.dms_input)
    return DMS_phenotype_name, dms_input, dms_output, msa_path, mutant_col, sequence


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print("args:", args)
    start_time = time.time()
    main(args)
    print("Done in " + str(time.time() - start_time) + " seconds")