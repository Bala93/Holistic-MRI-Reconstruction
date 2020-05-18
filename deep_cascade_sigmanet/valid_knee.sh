MODEL='sigmanet-deep-cascade'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='knee-mc-coronal-pd'


#<<ACC_FACTOR_4x
BATCH_SIZE=1
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_4x/'${MODEL}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_4x/'${MODEL}'/results'
DEVICE='cuda:0'

DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/acc_4x/validation/'

python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH}
#ACC_FACTOR_4x

