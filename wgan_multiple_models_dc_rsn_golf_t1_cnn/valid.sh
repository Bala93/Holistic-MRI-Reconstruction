BASE_PATH='/data/balamurali'
ACCELERATION='4' #acceleration factor
DATASET_TYPE='mrbrain_flair'
EPOCH='22'
MODE='vif' #vif
USMASK_DIR=${BASE_PATH}'/us_masks/'${DATASET_TYPE}
VAL_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/cartesian/validation/acc_'${ACCELERATION}'x'
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/cartesian/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/model_'${EPOCH}'.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/cartesian/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}
SEG_UNET_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/cartesian/acc_5x/rsnlossfunctions/unetLEM/best_model.pt'
DNCN_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/dc3-rsn-assist'
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --val-path ${VAL_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_DIR} --acceleration ${ACCELERATION} --seg_unet_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH}
