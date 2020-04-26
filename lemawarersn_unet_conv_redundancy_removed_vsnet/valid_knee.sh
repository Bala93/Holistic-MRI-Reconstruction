#MODEL='vs-net-recon-synergy-net'
MODEL='lemawarersn_unet_conv_redundancy_removed_vsnet'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='knee-mc-coronal-pd'

#<<ACC_FACTOR_4x
ACC_FACTOR='4x'
BATCH_SIZE=1
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
DEVICE='cuda:0'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/validation/'
SEG_UNET_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_5x/rsnlossfunctions/unetLEM/best_model.pt' # check this with sriprabha
DNCN_MODEL_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/vs-net-recon-synergy-net/'


echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --seg_unet_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_MODEL_PATH} 
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --seg_unet_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_MODEL_PATH} 

#ACC_FACTOR_4x

