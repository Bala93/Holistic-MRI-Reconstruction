EXP_NAME='dc_rsn'
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:1'
EXP_DIR='/media/htic/NewVolume2/balamurali/fastmri_experiments/'${EXP_NAME}
TRAIN_PATH='/media/htic/NewVolume5/fastmri/singlecoil_train/'
VALIDATION_PATH='/media/htic/NewVolume5/fastmri/singlecoil_val/'
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} 
