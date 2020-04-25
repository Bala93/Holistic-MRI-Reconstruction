MODEL='dc3-rsn'
BASE_PATH='/data/balamurali'
DATASET_TYPE='mrbrain_flair'

ACC_FACTOR='8x'
BATCH_SIZE=1
DEVICE='cuda:0'

CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
