BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
ACCELERATION='4' #acceleration factor
DATASET_TYPE='knee-mc-coronal-pd'
EPOCH='41'
MODE='psnr' #vif
VAL_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/validation'
CHECKPOINT=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/model_'${EPOCH}'.pt'
OUT_DIR=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --val-path ${VAL_PATH} --acceleration ${ACCELERATION}
