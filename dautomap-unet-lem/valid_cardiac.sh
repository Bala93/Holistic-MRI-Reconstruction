MODEL='dautomap-cnn-three-channel-lem'
DATASET_TYPE='cardiac'

<<ACC_FACTOR_2x
ACC_FACTOR='2x'
CHECKPOINT='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
ACC_FACTOR_2x

<<ACC_FACTOR_2.5x
ACC_FACTOR='2_5x'
CHECKPOINT='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
ACC_FACTOR_2.5x


<<ACC_FACTOR_3_3x
ACC_FACTOR='3_3x'
CHECKPOINT='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'


echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
ACC_FACTOR_3_3x


<<ACC_FACTOR_4x
ACC_FACTOR='4x'
CHECKPOINT='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:1'
DATA_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_4x/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_4x/dautomap/best_model.pt'


echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
ACC_FACTOR_4x


#<<ACC_FACTOR_5x
ACC_FACTOR='5x'
CHECKPOINT='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'
SEG_UNET_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/unetLEM/best_model.pt'
DNCN_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/deep-cascade-cnn/best_model.pt'
USMASK_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/Reconstruction-for-MRI/us_masks/'${DATASET_TYPE}'/'${MASK_TYPE}

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH} --seg_unet_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_MODEL_PATH} --usmask_path ${USMASK_PATH}
#ACC_FACTOR_5x


<<ACC_FACTOR_8x
ACC_FACTOR='8x'
CHECKPOINT='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'


echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
ACC_FACTOR_8x

