EXP_NAME='vs_rsn_csv'
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
EXP_DIR='/media/htic/NewVolume2/balamurali/fastmri_experiments/'${EXP_NAME}
TRAIN_PATH='/media/htic/NewVolume5/fastmri/multicoil_train/'
VALIDATION_PATH='/media/htic/hd/fastmri/multicoil_valid/'
TRAIN_CSV_PATH='/media/htic/NewVolume3/Balamurali/fastmri/multicoil_train.csv'
VALIDATION_CSV_PATH='/media/htic/NewVolume3/Balamurali/fastmri/multicoil_valid.csv'
CHALLENGE='multicoil'

python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --train-csv-path ${TRAIN_CSV_PATH} --valid-csv-path ${VALIDATION_CSV_PATH} --challenge ${CHALLENGE} --report-interval 1 --data-parallel

