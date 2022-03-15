# Basically a copy of the first part of run_svi.py, but not loading the model after calculating the weights
import time
import sys

sys.path.insert(0, "../DeepSequence/")
import numpy as np

import helper
import argparse

parser = argparse.ArgumentParser(description="Calculating the weights and storing in weights_dir.")
parser.add_argument("--dataset", type=str, default="BLAT_ECOLX",
                    help="Dataset name for fitting model.")
parser.add_argument("--theta-override", type=float, default=None,
                    help="Override the model theta.")
# Keeping this different from weights_dir just so that we don't make mistakes and overwrite weights
parser.add_argument("--alignments_dir", type=str, help="Overrides the default ./datasets/alignments/")
parser.add_argument("--weights_dir_out", type=str, default="", help="Location to store weights.")
args = parser.parse_args()

# DataHelper expects the dataset name without extension
args.dataset = args.dataset.split(".a2m")[0]
assert not args.dataset.endswith(".a2m")

data_params = {
    "dataset": args.dataset,
    "weights_dir": args.weights_dir_out,
}

if __name__ == "__main__":
    start_time = time.time()

    data_helper = helper.DataHelper(dataset=data_params["dataset"],
                                    working_dir='.',
                                    theta=args.theta_override,
                                    weights_dir=data_params["weights_dir"],
                                    calc_weights=True,
                                    alignments_dir=args.alignments_dir,
                                    save_weights=True,
                                    )
    # write out what theta was used
    data_params['theta'] = data_helper.theta

    end_time = time.time()
    print("Done in " + str(time.time() - start_time) + " seconds")
