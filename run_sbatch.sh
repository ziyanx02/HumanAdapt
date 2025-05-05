#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH -t 360
#SBATCH -p gpu-preempt
#SBATCH --constraint='1080ti|2080ti|a100'

set -e
eval "$(conda shell.bash hook)"
conda activate AcPPO

TASK_NAME=$1
shift

echo "Running task: ${TASK_NAME} on $(hostname)"

nvidia-smi

python run.py -t "${TASK_NAME}" "$@"
