#MODEL='vs-net-recon-synergy-net'
MODEL='lemawarersn_unet_conv_redundancy_removed_vsnet'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
#DATASET_TYPE='knee-mc-axial-t2'
DATASET_TYPE='knee-mc-coronal-pd'

#<<ACC_FACTOR_4x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='4x'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/train/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/validation/'
SEG_UNET_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_5x/rsnlossfunctions/unetLEM/best_model.pt' # check this with sriprabha
DNCN_MODEL_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/vs-net-recon-synergy-net/'

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --seg_unet_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_MODEL_PATH} 
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --seg_unet_path ${SEG_UNET_PATH} --dncn_model_path ${DNCN_MODEL_PATH} 

#ACC_FACTOR_4x

