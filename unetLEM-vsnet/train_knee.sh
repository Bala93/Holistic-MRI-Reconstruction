MODEL='unetLEM'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
#DATASET_TYPE='knee-mc-coronal-pd'
DATASET_TYPE='knee-mc-coronal-pd-fs'
#MASK_TYPE='gaussian'
#MASK_TYPE='cartesian'

#<<ACC_FACTOR_5x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:1'
ACC_FACTOR='5x'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/train/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/validation/'

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
#--usmask_path ${USMASK_PATH}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
#ACC_FACTOR_5x


