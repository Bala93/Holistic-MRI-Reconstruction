MODEL='lemrsn-redundancyremoved_AssistLatDec-unetlem-relu-noRe512-96-t1assist'
BASE_PATH='/data/balamurali'
#BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='mrbrain_flair'
#DATASET_TYPE='cardiac'
#MASK_TYPE='gaussian'
MASK_TYPE='cartesian'

<<ACC_FACTOR_4x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='4x'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
USMASK_PATH=${BASE_PATH}'/us_masks/'${DATASET_TYPE}
#SEG_UNET_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-UnetLEMEdgeloss/best_model.pt'
SEG_UNET_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_5x/rsnlossfunctions/unetLEM/best_model.pt'
#DNCN_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-L1loss/best_model.pt'
DNCN_PATH='/data/balamurali/experiments/mrbrain_flair/acc_4x/dc3-rsn-assist'
echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --seg_unet_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH}
ACC_FACTOR_4x



<<ACC_FACTOR_5x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='5x'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
USMASK_PATH=${BASE_PATH}'/us_masks/'${DATASET_TYPE}
#SEG_UNET_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-UnetLEMEdgeloss/best_model.pt'
SEG_UNET_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_5x/rsnlossfunctions/unetLEM/best_model.pt'
#DNCN_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-L1loss/best_model.pt'
DNCN_PATH='/data/balamurali/experiments/mrbrain_flair/acc_5x/dc3-rsn-assist'
echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --seg_unet_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH}
ACC_FACTOR_5x



#<<ACC_FACTOR_8x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='8x'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
USMASK_PATH=${BASE_PATH}'/us_masks/'${DATASET_TYPE}
#SEG_UNET_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-UnetLEMEdgeloss/best_model.pt'
SEG_UNET_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_5x/rsnlossfunctions/unetLEM/best_model.pt'
#DNCN_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/experiments/mrbrain_t1/cartesian/acc_5x/rsnlossfunctions/dc-cnn-L1loss/best_model.pt'
DNCN_PATH='/data/balamurali/experiments/mrbrain_flair/acc_8x/dc3-rsn-assist'
echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --seg_unet_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_PATH}
#ACC_FACTOR_8x


