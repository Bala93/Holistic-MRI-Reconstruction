BASE_PATH='/data/balamurali'
ACCELERATION='4' #acceleration factor
DATASET_TYPE='mrbrain_flair'
MODE='psnr' #vif

echo REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/cartesian/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}'/report.txt'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/cartesian/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}'/report.txt'
cat ${REPORT_PATH}
echo "\n"

MODE='vif' #vif

echo REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/cartesian/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}'/report.txt'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/cartesian/acc_'${ACCELERATION}'x/wgan_multiple_models_dcrsn/results_'${MODE}'/report.txt'
cat ${REPORT_PATH}
echo "\n"

