MODEL='dautomap-cnn-three-channel-lem'
DATASET_TYPE='cardiac'

<<ACC_FACTOR_2x
BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='2x'

EXP_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}
TRAIN_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'

USMASK_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/Reconstruction-for-MRI/us_masks/'${DATASET_TYPE}'/'${MASK_TYPE}

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}

ACC_FACTOR_2x


<<ACC_FACTOR_2.5x
BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='2_5x'

EXP_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}
TRAIN_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}

ACC_FACTOR_2.5x

<<ACC_FACTOR_3_3x
BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='3_3x'

EXP_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}
TRAIN_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}

ACC_FACTOR_3_3x


<<ACC_FACTOR_4x
BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:1'
ACC_FACTOR='4x'

EXP_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}
TRAIN_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
ACC_FACTOR_4x


#<<ACC_FACTOR_5x
BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='5x'

EXP_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}
TRAIN_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'
SEG_UNET_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/unetLEM/best_model.pt'
DNCN_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/deep-cascade-cnn/best_model.pt'
USMASK_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/Reconstruction-for-MRI/us_masks/'${DATASET_TYPE}'/'${MASK_TYPE}

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH} --seg_unet_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_MODEL_PATH} --usmask_path ${USMASK_PATH}
#ACC_FACTOR_5x


<<ACC_FACTOR_8x

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='8x'

EXP_DIR='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}
TRAIN_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}

UNET_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/unet/best_model.pt'
dAUTOMAP_MODEL_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/cardiac/acc_'${ACC_FACTOR}'/dautomap/best_model.pt'

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --unet_model_path ${UNET_MODEL_PATH} --dautomap_model_path ${dAUTOMAP_MODEL_PATH}

ACC_FACTOR_8x

