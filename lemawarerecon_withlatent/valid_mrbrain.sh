MODEL='dc-cnn-featwithlatent'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='mrbrain_t1'
#MASK_TYPE='gaussian'
MASK_TYPE='cartesian'


#<<ACC_FACTOR_5x
ACC_FACTOR='5x'
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:1'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
SEG_UNET_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-UnetLEMEdgeloss/best_model.pt'
DNCN_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-L1loss/best_model.pt'

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH} 
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH} 
#ACC_FACTOR_5x



