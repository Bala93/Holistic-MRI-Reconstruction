EXP_NAME='dc_rsn_csv'
BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
EXP_DIR='/media/htic/NewVolume2/balamurali/fastmri_experiments/'${EXP_NAME}
TRAIN_PATH='/media/htic/NewVolume5/fastmri/singlecoil_train/'
VALIDATION_PATH='/media/htic/NewVolume5/fastmri/singlecoil_val/'
TRAIN_CSV_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/Holistic-MRI-Reconstruction/dc_rsn_fastmri/train_batch_randomized.csv'
VALIDATION_CSV_PATH='/media/htic/NewVolume1/murali/MR_reconstruction/Holistic-MRI-Reconstruction/dc_rsn_fastmri/valid_batch_randomized.csv'

python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --train-csv-path ${TRAIN_CSV_PATH} --valid-csv-path ${VALIDATION_CSV_PATH} --data-parallel

