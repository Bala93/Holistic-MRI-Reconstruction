MODEL='dagan' 
DATASET_TYPE='kirby'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'


#<<ACC_FACTOR_4x
ACC_FACTOR='4x'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${REPORT_PATH}
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_4x

