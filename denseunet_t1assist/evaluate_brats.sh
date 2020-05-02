MODEL='unet-assist'
#MODEL='zf'
BASE_PATH='/data/balamurali'
DATASET_TYPE='brats'

ACC_FACTOR='5x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/'
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MODEL}'/results'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MODEL}'/'
echo python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
