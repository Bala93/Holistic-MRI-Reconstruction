#MODEL='dc-rsn-feat-redundancyremoved_AssistEverywhere-unetlem-relu-trial2'
#MODEL='dc-rsn-feat-redundancyremoved_AssistEverywhere-unetlem-relu-trial2-nolemimg512'

MODEL='lemrsn-redundancyremoved_AssistLatDec-unetlem-relu-nolemimg512-t1assist'
#MODEL='lemrsn-redundancyremoved_AssistLatDec-unetlem-relu-noRe512-96-t1assist'
#MODEL='firstblocklemrsn-redundancyremoved_AssistLatDec-unetlem-relu-noRe512-96-t1assist'
BASE_PATH='/data/balamurali'
DATASET_TYPE='mrbrain_flair'
#MASK_TYPE='gaussian'
MASK_TYPE='cartesian'

#<<ACC_FACTOR_4x
ACC_FACTOR='4x'
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
#SEG_UNET_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-UnetLEMEdgeloss/best_model.pt'

USMASK_PATH=${BASE_PATH}'/us_masks/'${DATASET_TYPE}
#SEG_UNET_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/unetLEM/best_model.pt'

SEG_UNET_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_5x/rsnlossfunctions/unetLEM/best_model.pt'
#DNCN_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-L1loss/best_model.pt'
DNCN_PATH='/data/balamurali/experiments/mrbrain_flair/acc_4x/dc3-rsn-assist'

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH} 
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH} --usmask_path ${USMASK_PATH} 
#ACC_FACTOR_4x



#<<ACC_FACTOR_5x
ACC_FACTOR='5x'
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
#SEG_UNET_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-UnetLEMEdgeloss/best_model.pt'

USMASK_PATH=${BASE_PATH}'/us_masks/'${DATASET_TYPE}
SEG_UNET_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/unetLEM/best_model.pt'
#DNCN_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-L1loss/best_model.pt'
DNCN_PATH='/data/balamurali/experiments/mrbrain_flair/acc_5x/dc3-rsn-assist'

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH} 
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH} --usmask_path ${USMASK_PATH} 
#ACC_FACTOR_5x


<<ACC_FACTOR_8x
ACC_FACTOR='8x'
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
#SEG_UNET_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-UnetLEMEdgeloss/best_model.pt'

USMASK_PATH=${BASE_PATH}'/us_masks/'${DATASET_TYPE}
SEG_UNET_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_5x/rsnlossfunctions/unetLEM/best_model.pt'
#DNCN_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-L1loss/best_model.pt'
DNCN_PATH='/data/balamurali/experiments/mrbrain_flair/acc_8x/dc3-rsn-assist'

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH} 
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH} --usmask_path ${USMASK_PATH} 
ACC_FACTOR_8x




