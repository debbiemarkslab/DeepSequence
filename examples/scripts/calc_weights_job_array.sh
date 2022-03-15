#!/bin/bash
#SBATCH -c 2                              # Request one core
#SBATCH -N 1                              # Request one node (if you request more than one core with -c, also using
                                          # -N 1 means all cores will be on the same node)
#SBATCH -t 0-5:59                         # Runtime in D-HH:MM format
#SBATCH -p short                          # Partition to run in
#SBATCH --mem=10G                         # Memory total in MB (for all cores)

# To get email notifications, set both of these options below
##SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
##SBATCH --mail-user="<your_email>@hms.harvard.edu"

#SBATCH --job-name="deepseq_calcweights_date"
# Job array-specific
#SBATCH --output=slurm-%A_%a-%x.out  # File to which STDOUT + STDERR will be written, %A: jobID, %a: array task ID, %x: jobname
#SBATCH --array=0-41%10  		  # Job arrays (e.g. 1-100 with a maximum of 5 jobs at once)

hostname
pwd
module load gcc/6.2.0 cuda/9.0
export THEANO_FLAGS='floatX=float32,device=cuda,force_device=True' # Otherwise will only raise a warning and carry on with CPU

# To generate this file from a directory, just do e.g. 'ls -1 ALIGNMENTS_DIR/*.a2m > msas.txt'
lines=( $(cat "msas.txt") )
dataset_name=${lines[$SLURM_ARRAY_TASK_ID]}
echo $dataset_name

## Monitor GPU usage (store outputs in ./gpu_logs/)
#/home/lov701/job_gpu_monitor.sh gpu_logs &

srun stdbuf -oL -eL /n/groups/marks/users/aaron/deep_seqs/deep_seqs_env/bin/python \
  /n/groups/marks/users/lood/DeepSequence_runs/examples/calc_weights.py \
  --dataset $dataset_name \
  --weights_dir_out /n/groups/marks/users/lood/DeepSequence_runs/weights_2021_11_16/ \
  --alignments_dir datasets/alignments/
#  --theta-override 0.9
