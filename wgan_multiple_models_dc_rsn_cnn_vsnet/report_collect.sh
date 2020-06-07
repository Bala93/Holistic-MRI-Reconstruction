BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
ACCELERATION='4' #acceleration factor
DATASET_TYPE='knee-mc-coronal-pd'
MODE='psnr' #vif

echo REPORT_PATH=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}'/report.txt'
REPORT_PATH=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}'/report.txt'
cat ${REPORT_PATH}
echo "\n"

MODE='vif' #vif

echo REPORT_PATH=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}'/report.txt'
REPORT_PATH=${BASE_PATH}'/experiments_reconglgan_setup/'${DATASET_TYPE}'/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}'/report.txt'
cat ${REPORT_PATH}
echo "\n"

