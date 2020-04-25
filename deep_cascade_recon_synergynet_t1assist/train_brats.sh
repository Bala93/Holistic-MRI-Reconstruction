MODEL='deep-cascade-rsn1-assist'
BASE_PATH='/data/balamurali'
DATASET_TYPE='brats'

BATCH_SIZE=8
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='5x'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/train/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/'
USMASK_PATH=${BASE_PATH}'/us_masks/'${DATASET_TYPE}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH}
