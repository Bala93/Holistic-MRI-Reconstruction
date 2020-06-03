BASE_PATH='/media/hticpose/drive2/Balamurali'
ACCELERATION='5'
DATASET_TYPE='cardiac'
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/train/acc_'${ACCELERATION}'x'
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACCELERATION}'x'
EXP_DIR=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn'
USMASK_DIR=${BASE_PATH}'/Reconstruction-for-MRI/us_masks/'${DATASET_TYPE}
DCRSNPATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/deep_cascade_recon_synergy_net/best_model.pt'
echo python train.py --train-path ${TRAIN_PATH} --val-path ${VALIDATION_PATH} --exp-dir ${EXP_DIR} --acceleration ${ACCELERATION} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_DIR} --dcrsnpath ${DCRSNPATH}
python train.py --train-path ${TRAIN_PATH} --val-path ${VALIDATION_PATH} --exp-dir ${EXP_DIR} --acceleration ${ACCELERATION} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_DIR} --dcrsnpath ${DCRSNPATH}
