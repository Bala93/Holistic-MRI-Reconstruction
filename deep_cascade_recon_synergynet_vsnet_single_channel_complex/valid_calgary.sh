MODEL='vs-net-rsn'
BASE_PATH='/data/balamurali'
DATASET_TYPE='calgary'


#<<ACC_FACTOR_5x
BATCH_SIZE=1
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MODEL}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MODEL}'/results'
DEVICE='cuda:0'

DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_4x'

#python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH}
echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH}
#ACC_FACTOR_5x

