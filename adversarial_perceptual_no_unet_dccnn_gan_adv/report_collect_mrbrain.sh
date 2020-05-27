
MODEL='deep-cascade-generator-unet-discriminator'
BASE_PATH='/media/hticpose/drive2/Balamurali'
DATASET_TYPE='mrbrain'

<<ACC_FACTOR_4x
ACC_FACTOR='4x'
REPORT_PATH=${BASE_PATH}'/experiments_reconseg/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
ACC_FACTOR_4x

#<<ACC_FACTOR_8x
ACC_FACTOR='8x'
REPORT_PATH=${BASE_PATH}'/experiments_reconseg/'${DATASET_TYPE}'/reconstruction/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_4x

