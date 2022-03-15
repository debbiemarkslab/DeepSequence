#!/bin/bash
#SBATCH -c 2                               # Request two cores
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 0-23:59                         # Runtime in D-HH:MM format
#SBATCH -p gpu_quad    #,gpu_marks,gpu,gpu_requeue        # Partition to run in
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

##SBATCH -o slurm_files/slurm-%j.out                 # File to which STDOUT + STDERR will be written, including job ID in filename
#SBATCH --job-name="deepseq_training"

# Job array-specific
#SBATCH --output=slurm_files/slurm-lvn-%A_%3a-%x.out   # Nice tip: using %3a to pad to 3 characters (23 -> 023)
##SBATCH --error=slurm_files/slurm-lvn-%A_%3a-%x.err   # Optional: Redirect STDERR to its own file
#SBATCH --array=0-49,100-149,200-249,300-349,400-449%10  # 5 seeds, e.g. 50 MSAs, with maximum 10 simultaneous jobs
#SBATCH --hold  # Holds job so that we can first manually check a few

# Quite neat workflow:
# Submit job array in held state, then release first job to test
# Add a dependency so that the next jobs are submitted as soon as the first job completes successfully:
# scontrol update Dependency=afterok:<jobid>_0 JobId=<jobid>
# Release all the other jobs; they'll be stuck until the first job is done
################################################################################

set -e # fail fully on first line failure (from Joost slurm_for_ml)

# Note: Remember to clear ~/.theano cache before running this script

echo "hostname: $(hostname)"
echo "Running from: $(pwd)"
echo "GPU available: $(nvidia-smi)"
module load gcc/6.2.0 cuda/9.0
export THEANO_FLAGS='floatX=float32,device=cuda,force_device=True' # Otherwise will only raise a warning and carry on with CPU

# To generate this file from a directory, just do e.g. '(cd ALIGNMENTS_DIR && ls -1 *.a2m) > datasets.txt'
lines=( $(cat "msa.txt") ) # v5 benchmark
DATASET_ID=$(($SLURM_ARRAY_TASK_ID % 100))  # Group a run of datasets together
seed_id=$(($SLURM_ARRAY_TASK_ID / 100))
seed_id=$SLURM_ARRAY_TASK_ID
seeds=(1 2 3 4 5)  # For some reason Theano won't accept seed 0..
SEED=${seeds[$seed_id]}
echo "DATASET_ID: $DATASET_ID, seed: $SEED"

dataset_name=${lines[$DATASET_ID]}
echo "dataset name: $dataset_name"

export WEIGHTS_DIR=weights_msa_tkmer_20220227
export ALIGNMENTS_DIR=msa_tkmer_20220227

## Monitor GPU usage (store outputs in ./gpu_logs/)
#/home/lov701/job_gpu_monitor.sh --interval 1m gpu_logs &

srun stdbuf -oL -eL /n/groups/marks/users/aaron/deep_seqs/deep_seqs_env/bin/python \
  /n/groups/marks/users/lood/DeepSequence_runs/run_svi.py \
  --dataset $dataset_name \
  --weights_dir $WEIGHTS_DIR \
  --alignments_dir $ALIGNMENTS_DIR \
  --seed $SEED
#  --theta-override 0.9
