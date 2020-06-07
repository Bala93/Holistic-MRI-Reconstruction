BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
ACCELERATION='5'
DATASET_TYPE='knee-mc-coronal-pd'
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/train'
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/validation'
EXP_DIR=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn'
DCRSNPATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/vs-net-recon-synergy-net/best_model.pt'
echo python train.py --train-path ${TRAIN_PATH} --val-path ${VALIDATION_PATH} --exp-dir ${EXP_DIR} --dcrsnpath ${DCRSNPATH}
python train.py --train-path ${TRAIN_PATH} --val-path ${VALIDATION_PATH} --exp-dir ${EXP_DIR} --dcrsnpath ${DCRSNPATH}
