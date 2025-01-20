#!/bin/bash
#SBATCH --job-name=leniabreeder
#SBATCH --output=leniabreeder_%A_%a.out
#SBATCH --error=leniabreeder_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a5000-test-q
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=1

# Load required modules
module purge
module load apptainer/1.1.9
module load cuda12.4/toolkit/12.4.1
module load cudnn9.3-cuda12.4/9.3.0.75

# Set up environment variables
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Create working directory
WORKDIR="run_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p ${WORKDIR}
cd ${WORKDIR}

# Build container if it doesn't exist
if [ ! -f ../leniabreeder.sif ]; then
    cd ../apptainer
    
    # Build container with conda base
    APPTAINER_NOHTTPS=1 apptainer build --sandbox ../leniabreeder.sif container.def
    
    if [ $? -ne 0 ]; then
        echo "Container build failed!"
        exit 1
    fi
    
    cd ../${WORKDIR}
fi

# Run the experiment
apptainer run --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    ../leniabreeder.sif \
    qd=aurora

apptainer run --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env PYTHONPATH="/workspace/src" \
    ../leniabreeder.sif \
    python /workspace/src/analysis/visualize_aurora.py "${PWD}"

# Copy results if needed
mkdir -p ../results/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
if [ -d "output" ]; then
    cp -r output/* ../results/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
fi

if [ -d "visualization" ]; then
    cp -r visualization/* ../results/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/
fi

# Clean up
cd ..
rm -rf ${WORKDIR}