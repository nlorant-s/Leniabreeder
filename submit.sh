#!/bin/bash
#SBATCH --job-name=leniabreeder
#SBATCH --output=leniabreeder_%A_%a.out
#SBATCH --error=leniabreeder_%A_%a.err
#SBATCH --time=07:29:00
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=1-51

# Load required modules
module purge
module load apptainer/1.1.9
module load cuda12.4/toolkit/12.4.1
module load cudnn9.3-cuda12.4/9.3.0.75

# Set up environment variables
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Create working directory and copy configs
WORKDIR="run_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p ${WORKDIR}
cd ${WORKDIR}

# Build container with conda base
cd ../apptainer
APPTAINER_NOHTTPS=1 apptainer build --force ../leniabreeder.sif container.def

cd ../${WORKDIR}

# Run the experiment
apptainer run --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --bind ${PWD}:/workspace/run \
    --bind ${HOME}/Leniabreeder:/workspace/leniabreeder \
    ../leniabreeder.sif \
    qd=aurora seed=$RANDOM

# Copy results if needed
mkdir -p ../results/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
if [ -d "output" ]; then
    cp -r output/* ../results/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
fi

# Clean up
cd ..
rm -rf ${WORKDIR}
