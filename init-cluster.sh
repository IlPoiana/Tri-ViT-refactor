# TO REMOVE just for visualization
export SAMPLE_NAME="IXI026-Guys-0696-T1.nii.gz" # "IXI012-HH-1211-T1.nii.gz"
export DATA_PATH="$(pwd)/data"

export IXI_TRAIN_PREPROCESSED="${DATA_PATH}/IXI_train_pre/preprocessed"
export IXI_EVAL_PREPROCESSED="${DATA_PATH}/IXI_validate_pre/preprocessed"
export IXI_TEST_PREPROCESSED="${DATA_PATH}/IXI_test_pre/preprocessed"
export SAMPLE="${DATA_PATH}/IXI_test_pre/voxel/${SAMPLE_NAME}"
