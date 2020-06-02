TRAIN_PATH='/media/hticpose/drive2/Balamurali/datasets/cardiac_1/train/acc_4x'
VALIDATION_PATH='/media/hticpose/drive2/Balamurali/datasets/cardiac_1/validation/acc_4x'
EXP_DIR='/media/hticpose/drive2/Balamurali/experiments_reconglgan_setup/wgan_multiple_models_dcrsn' # folder to save models and write summary
ACCELERATION=4
python train.py --train-path ${TRAIN_PATH} --val-path ${VALIDATION_PATH} --exp-dir ${EXP_DIR} --acceleration ${ACCELERATION}
