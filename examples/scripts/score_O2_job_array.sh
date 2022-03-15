#!/bin/bash
# This script can be used to score an ensemble of models.
# To score just one, just set a constant seed e.g. seeds=(42) and only use array ids between 0-99.
#SBATCH -c 2                           	# Request one core
#SBATCH -N 1                           	# Request one node (if you request more than one core with -c, also using
                                       	# -N 1 means all cores will be on the same node)
#SBATCH -t 0-23:59                      # Runtime in D-HH:MM format
#SBATCH -p gpu_quad,gpu_marks,gpu #,gpu_requeue        # Partition to run in
# If on gpu_quad, use teslaV100s
# If on gpu_requeue, use teslaM40 or a100?
# If on gpu, any of them are fine (teslaV100, teslaM40, teslaK80) although K80 sometimes is too slow
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_doublep          # Only use double precision GPUs, otherwise our theano version can't use them
#SBATCH --qos=gpuquad_qos
#SBATCH --mem=20G                          # Memory total in MB (for all cores)

# To use email notifications, set both of the following options
##SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
##SBATCH --mail-user="<email>"

#SBATCH -vv  # Verbose

##SBATCH -o slurm-%j.out                 # File to which STDOUT + STDERR will be written, including job ID in filename
#SBATCH --job-name="score_deepseq"
# Job array-specific
#SBATCH --output=slurm_files/slurm-lvn-%A_%3a-%x.out        # Nice tip: using %3a to pad to 3 characters (23 -> 023)
##SBATCH --array=0-86,100-186,200-286,300-386,400-486%10          		# 87 DMSs in total benchmark

################################################################################

set -e # fail fully on first line failure (from Joost slurm_for_ml)

# Note: Remember to clear ~/.theano cache before running this script, otherwise jobs eventually start crashing while compiling theano simultaneously

echo "hostname: $(hostname)"
echo "Running from: $(pwd)"
echo "GPU available: $(nvidia-smi)"
module load gcc/6.2.0 cuda/9.0
export THEANO_FLAGS='floatX=float32,device=cuda,force_device=True,traceback.limit=20, exception_verbosity=high' # Otherwise will only raise a warning and carry on with CPU

DATASET_ID=$(($SLURM_ARRAY_TASK_ID % 100))  # Group all datasets together in 0xx, 1xx, 2xx, etc.
SEED_ID=$(($SLURM_ARRAY_TASK_ID / 100))
seeds=(1 2 3 4 5)  # For some reason Theano won't accept SEED 0..
SEED=${seeds[$SEED_ID]}
echo "DATASET_ID: $DATASET_ID, SEED: $SEED"

export dms_mapping=#CSV containing MSA and DMS mappings
export dms_input_folder=#directory containing DMS input files
# Remember to create this folder before run:
export dms_output_folder=#directory to store output CSVs
export msa_path=#Folder containing MSA files
export model_checkpoint_dir=#Folder containing model checkpoints

# Monitor GPU usage (store outputs in ./gpu_logs/)
#/home/lov701/job_gpu_monitor.sh --interval 1m gpu_logs &

srun stdbuf -oL -eL /n/groups/marks/users/aaron/deep_seqs/deep_seqs_env/bin/python \
  /n/groups/marks/users/lood/DeepSequence_runs/run_muteff_pred_seqs_batch.py \
  --dms_mapping $dms_mapping \
  --dms_input_dir $dms_input_folder \
  --dms_output_dir $dms_output_folder \
  --msa_path $msa_path \
  --model_checkpoint $model_checkpoint_dir \
  --dms_index $DATASET_ID \
  --samples 2000 \
  --batch_size 8000 \
  --seed "$SEED"
#  --theta-override 0.9


