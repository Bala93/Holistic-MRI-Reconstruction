BASE_PATH='/data/balamurali'
ACCELERATION='4'
DATASET_TYPE='mrbrain_flair'
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/cartesian/train/acc_'${ACCELERATION}'x'
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/cartesian/validation/acc_'${ACCELERATION}'x'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/cartesian/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn'
USMASK_DIR=${BASE_PATH}'/us_masks/'${DATASET_TYPE}
DCRSNPATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/cartesian/acc_4x/rsnlossfunctions/lemrsn-redundancyremoved_AssistLatDec-unetlem-relu-noRe512-t1assist/best_model.pt'
SEG_UNET_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/cartesian/acc_5x/rsnlossfunctions/unetLEM/best_model.pt'
DNCN_PATH='/data/balamurali/experiments/mrbrain_flair/acc_4x/dc3-rsn-assist'

echo python train.py --train-path ${TRAIN_PATH} --val-path ${VALIDATION_PATH} --exp-dir ${EXP_DIR} --acceleration ${ACCELERATION} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_DIR} --dcrsnpath ${DCRSNPATH} --seg_unet_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH}
python train.py --train-path ${TRAIN_PATH} --val-path ${VALIDATION_PATH} --exp-dir ${EXP_DIR} --acceleration ${ACCELERATION} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_DIR} --dcrsnpath ${DCRSNPATH} --seg_unet_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH}
