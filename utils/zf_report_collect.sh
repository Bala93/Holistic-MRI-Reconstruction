DATASET_TYPE='kirby90'
MODEL='zf'
ACC_FACTOR='5x'
BASE_PATH='/data/balamurali'
MASK_TYPE=${1}

REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
