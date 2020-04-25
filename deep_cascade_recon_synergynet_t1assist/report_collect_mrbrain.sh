
#MODEL='deep-cascade-rsn1'
#MODEL='zf'
BASE_PATH='/data/balamurali'
DATASET_TYPE='mrbrain_flair'

ACC_FACTOR='5x'

MODEL='dc3-rsn'
echo ${MODEL}
echo ${DATASET_TYPE}
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"

MODEL='dc3-rsn-assist'
echo ${MODEL}
echo ${DATASET_TYPE}
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"

