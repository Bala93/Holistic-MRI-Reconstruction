
MODEL='deep-cascade-densenet'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='kirby90'
MASK_TYPE='cartesian'

#<<ACC_FACTOR_4x
ACC_FACTOR='4x'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_4x

<<ACC_FACTOR_5x
ACC_FACTOR='5x'
echo ${ACC_FACTOR}
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
cat ${REPORT_PATH}
echo "\n"
ACC_FACTOR_5x

<<ACC_FACTOR_8x
ACC_FACTOR='8x'
echo ${ACC_FACTOR}
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
cat ${REPORT_PATH}
echo "\n"
ACC_FACTOR_8x
