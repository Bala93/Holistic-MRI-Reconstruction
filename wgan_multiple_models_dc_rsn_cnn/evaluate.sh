BASE_PATH='/media/hticpose/drive2/Balamurali'
ACCELERATION='4' #acceleration factor
DATASET_TYPE='cardiac'
MODE='vif' #vif

TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACCELERATION}'x'
PREDICTIONS_PATH=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}
REPORT_PATH=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 


