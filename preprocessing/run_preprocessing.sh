#!/usr/bin/env bash
echo run this inside "preprocessing" folder
 
# set -euo pipefail

IXI_TEST=../data/IXI_test

: ${IXI_DATASET:=$IXI_TEST} # variable expansion for preprocessing whatever directory is passed
OUT_ROOT=${IXI_DATASET}_pre

START_INDEX=${1:-0}     # default: 0
NUM_FILES=${2:-1}       # default: 1

export OMP_NUM_THREADS=8

mkdir -p ${OUT_ROOT}/bet
mkdir -p ${OUT_ROOT}/fast
mkdir -p ${OUT_ROOT}/fnirt
mkdir -p ${OUT_ROOT}/voxel
mkdir -p ${OUT_ROOT}/preprocessed

# Collect and sort files
mapfile -t FILES < <(ls ${IXI_DATASET}/*.nii.gz | sort)

TOTAL_FILES=${#FILES[@]}

echo "Total files available: ${TOTAL_FILES}"
echo "Processing from index ${START_INDEX}, count ${NUM_FILES}"

END_INDEX=$((START_INDEX + NUM_FILES))

if (( START_INDEX >= TOTAL_FILES )); then
    echo "Start index out of range"
    exit 1
fi

for ((i=START_INDEX; i<END_INDEX && i<TOTAL_FILES; i++)); do
    SAMPLE=${FILES[$i]}
    FILENAME=$(basename "$SAMPLE")
    SAMPLE_NAME=${FILENAME%.nii.gz}
    BRAIN_MASK=${OUT_ROOT}/bet/${SAMPLE_NAME}_mask.nii.gz

    echo "=============================="
    echo "Processing [$i]: $SAMPLE_NAME"
    echo "=============================="

    BET_OUT=${OUT_ROOT}/bet/${SAMPLE_NAME}
    FAST_OUT=${OUT_ROOT}/fast/${SAMPLE_NAME}
    FNIRT_OUT=${OUT_ROOT}/fnirt/${SAMPLE_NAME}
    FNIRT_MASK=${OUT_ROOT}/fnirt/${SAMPLE_NAME}_mask.nii.gz
    VOXEL_OUT=${OUT_ROOT}/voxel/${SAMPLE_NAME}
    INTERP_OUT=${OUT_ROOT}/preprocessed/${SAMPLE_NAME}

    echo "Running bet"
    bet "$SAMPLE" "$BET_OUT" -m

    echo "Running fast"
    fast -B -o "$FAST_OUT" "${BET_OUT}.nii.gz"

    echo "Running fnirt" # From the raw MRI scans to 1mm MNI space(not 2!!)
    fnirt \
        --in="${FAST_OUT}_restore.nii.gz" \
        --config=my_fnirt.cnf \
        --iout="$FNIRT_OUT" \
        --cout="${FNIRT_OUT}_warpcoef.nii.gz" 
    
    ## Warping the mask
    # echo "Warping the mask to fnirt space"
    # applywarp \
    #     --in="${BRAIN_MASK}" \
    #     --ref="${FNIRT_OUT}.nii.gz" \
    #     --warp="${FNIRT_OUT}_warpcoef.nii.gz" \
    #     --out="${FNIRT_MASK}" \
    #     --interp=nn
    ## ----

    echo "voxel norm"
    ## FNIRT interpolated mask 
    # mean_val=$(fslstats "${FNIRT_OUT}.nii.gz" -M -k "${FNIRT_MASK}") 
    # std_val=$(fslstats "${FNIRT_OUT}.nii.gz" -S -k "${FNIRT_MASK}")
    ## ----
    mean_val=$(fslstats "${FNIRT_OUT}.nii.gz" -M -k "${BRAIN_MASK}") 
    std_val=$(fslstats "${FNIRT_OUT}.nii.gz" -S -k "${BRAIN_MASK}")    
    fslmaths ${FNIRT_OUT}.nii.gz -div ${std_val} ${OUT_ROOT}/voxel/${SAMPLE_NAME}.nii.gz
    # echo "mean: $mean_val, std: $std_val"

    echo "interpolation back to 2mm"
    flirt -in ${VOXEL_OUT}.nii.gz -ref $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz -out ${INTERP_OUT}.nii.gz -applyisoxfm 2

    echo "Cleaning intermediate files"
    rm -f ${OUT_ROOT}/bet/*
    rm -f ${OUT_ROOT}/fast/*
    rm -f ${OUT_ROOT}/fnirt/*
    rm -f ${OUT_ROOT}/voxel/*

    echo "Finished $SAMPLE_NAME"
done

echo "All done"