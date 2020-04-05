
MODEL='mrn5-recon-synergy'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='cardiac'
#MASK_TYPE='gaussian'
MASK_TYPE='cartesian'

echo ${MODEL}
echo ${DATASET_TYPE}
echo ${MASK_TYPE}

#<<ACC_FACTOR_4x
ACC_FACTOR='4x'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_4x

#<<ACC_FACTOR_5x
ACC_FACTOR='5x'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_5x

