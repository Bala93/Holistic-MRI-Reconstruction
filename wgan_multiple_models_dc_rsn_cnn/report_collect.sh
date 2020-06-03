
BASE_PATH='/media/hticpose/drive2/Balamurali'
ACCELERATION='4' #acceleration factor
DATASET_TYPE='cardiac'
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

