BASE_PATH='/data/balamurali/'
ACCELERATION='4' #acceleration factor
DATASET_TYPE='mrbrain_flair'
MODE='vif' #vif

TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/cartesian/validation/acc_'${ACCELERATION}'x'
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/cartesian/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/cartesian/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 


