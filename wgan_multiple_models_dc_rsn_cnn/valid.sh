BASE_PATH='/media/hticpose/drive2/Balamurali'
ACCELERATION='4' #acceleration factor
DATASET_TYPE='cardiac'
EPOCH='149'
MODE='vif' #vif
USMASK_DIR=${BASE_PATH}'/Reconstruction-for-MRI/us_masks/'${DATASET_TYPE}
VAL_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACCELERATION}'x'
CHECKPOINT=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/model_'${EPOCH}'.pt'
OUT_DIR=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --val-path ${VAL_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_DIR} --acceleration ${ACCELERATION}
