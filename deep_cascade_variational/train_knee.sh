MODEL='variational-deep-cascade'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='knee-mc-coronal-pd-fs'

#<<ACC_FACTOR_4x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'

EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_4x/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/acc_4x/train/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/acc_4x/validation/'

python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} 

#ACC_FACTOR_4x

