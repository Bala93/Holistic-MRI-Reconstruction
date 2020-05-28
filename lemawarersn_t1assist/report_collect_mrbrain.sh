
#MODEL='dc-rsn-feat-redundancyremoved_AssistEverywhere-unetlem-relu-trial2'
#MODEL='dc-rsn-feat-redundancyremoved_AssistEverywhere-unetlem-relu-trial2-nolemimg512'
#MODEL='lemrsn-redundancyremoved_AssistLatDec-unetlem-relu-nolemimg512-t1assist'
MODEL='lemrsn-redundancyremoved_AssistLatDec-unetlem-relu-noRe512-t1assist'
#MODEL='dc-rsn-feat-redundancyremoved_AssistEverywhere-unetlem-relu'
BASE_PATH='/data/balamurali'
DATASET_TYPE='mrbrain_flair'
#MASK_TYPE='gaussian'
MASK_TYPE='cartesian'

echo ${MODEL}
echo ${DATASET_TYPE}
echo ${MASK_TYPE}

#<<ACC_FACTOR_4x
ACC_FACTOR='4x'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_4x

#<<ACC_FACTOR_5x
ACC_FACTOR='5x'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_5x

#<<ACC_FACTOR_8x
ACC_FACTOR='8x'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_8x



