MODEL='dautomap-cnn-three-channel'
DATASET_TYPE='kirby90'
MASK_TYPE='cartesian'


<<ACC_FACTOR_4x
ACC_FACTOR='4x'
CHECKPOINT='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:1'
DATA_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'


echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
ACC_FACTOR_4x


<<ACC_FACTOR_5x
ACC_FACTOR='5x'
CHECKPOINT='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}

ACC_FACTOR_5x


#<<ACC_FACTOR_8x
ACC_FACTOR='8x'

CHECKPOINT='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}

#ACC_FACTOR_8x

