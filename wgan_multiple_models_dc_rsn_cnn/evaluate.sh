TARGET_PATH='/media/hticpose/drive2/Balamurali/datasets/cardiac/validation/acc_4x'
PREDICTIONS_PATH='/media/hticpose/drive2/Balamurali/experiments_reconglgan_setup/wgan_multiple_models_dcrsn/results'
REPORT_PATH='/media/hticpose/drive2/Balamurali/experiments_reconglgan_setup/wgan_multiple_models_dcrsn/'
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 


