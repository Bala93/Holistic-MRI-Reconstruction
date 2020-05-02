
#MODEL='deep-cascade-rsn1'
#MODEL='zf'
BASE_PATH='/data/balamurali'
DATASET_TYPE='mrbrain_flair'

ACC_FACTOR='4x'

MODEL='dense-unet-assist'
echo ${MODEL}
echo ${DATASET_TYPE}
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${REPORT_PATH}
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"


ACC_FACTOR='5x'

MODEL='dense-unet-assist'
echo ${MODEL}
echo ${DATASET_TYPE}
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${REPORT_PATH}
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"



ACC_FACTOR='8x'

MODEL='dense-unet-assist'
echo ${MODEL}
echo ${DATASET_TYPE}
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${REPORT_PATH}
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"


