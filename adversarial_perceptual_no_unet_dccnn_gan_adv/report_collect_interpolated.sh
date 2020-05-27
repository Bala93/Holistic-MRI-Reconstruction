ALPHA=${1}
MODEL='adversarial_perceptuall_no_unet'
DATASET_TYPE='cardiac'
BASE_PATH='/media/hticpose/drive2/Balamurali/'

<<ACC_FACTOR_2x
ACC_FACTOR='2x'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${REPORT_PATH}
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
ACC_FACTOR_2x

#<<ACC_FACTOR_4x
ACC_FACTOR='4x'
REPORT_PATH=${BASE_PATH}'/experiments_reconseg/'${DATASET_TYPE}'/reconstruction/acc_'${ACC_FACTOR}'/'${MODEL}'/report_alpha_'${ALPHA}'.txt'
echo ${REPORT_PATH}
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_4x

<<ACC_FACTOR_8x
ACC_FACTOR='8x'
REPORT_PATH=${BASE_PATH}'/experiments_reconseg/'${DATASET_TYPE}'/reconstruction/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${REPORT_PATH}
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
ACC_FACTOR_8x

