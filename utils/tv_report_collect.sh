DATASET_TYPE='kirby90'
MODEL='tv'
ACC_FACTOR='4x'
BASE_PATH='/data/balamurali'
MASK_TYPE='cartesian'

REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
