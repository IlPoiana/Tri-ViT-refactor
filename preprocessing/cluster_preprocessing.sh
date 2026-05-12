#!/usr/bin/env bash
#SBATCH --job-name=tri-vit
#SBATCH --output=logs/preproc_%A_%a.out
#SBATCH --error=logs/preproc_%A_%a.err
#SBATCH  -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --partition=edu-long
#SBATCH --array=0-9

# Usage: sbatch sbatch_preprocessing.sh /path/to/IXI_dataset [total_files_per_job]

# Directory containing the data (passed as argument)
IXI_DATASET=${1:-"../data/IXI_test"}
echo "Processing dataset: ${IXI_DATASET}"

# Number of files to process per job (default: 1)
: ${FILES_PER_JOB:=1}
echo "Files per job: ${FILES_PER_JOB}"

# Calculate start index for this array job
START_INDEX=$((SLURM_ARRAY_TASK_ID * FILES_PER_JOB))

# Check if the number of elements to process exceeds the number of files available 
mapfile -t FILES < <(ls ${IXI_DATASET}/*.nii.gz | sort)

TOTAL_FILES=${#FILES[@]}
if (( (START_INDEX + FILES_PER_JOB) > TOTAL_FILES )); then
    FILES_PER_JOB=$((TOTAL_FILES - START_INDEX))
fi

# Create logs directory
mkdir -p logs

# Export environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Run preprocessing
# IXI_DATASET="${IXI_DATASET}" ./run_preprocessing.sh ${START_INDEX} ${FILES_PER_JOB} 
apptainer exec singularity-fsl_5.0.10.sif \
    bash -c "IXI_DATASET='${IXI_DATASET}' ./run_preprocessing.sh ${START_INDEX} ${FILES_PER_JOB}"

echo "Job ${SLURM_ARRAY_TASK_ID} completed: processed files starting at ${START_INDEX}"