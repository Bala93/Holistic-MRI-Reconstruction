BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
ACCELERATION='5' #acceleration factor
DATASET_TYPE='calgary'
MODE='psnr' #vif

TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACCELERATION}'x'
PREDICTIONS_PATH=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}
REPORT_PATH=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 


