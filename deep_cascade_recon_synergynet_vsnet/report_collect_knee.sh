MODEL='vs-net-recon-synergy-net'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='knee-mc'

echo ${MODEL}
echo ${DATASET_TYPE}
echo ${MASK_TYPE}

#<<ACC_FACTOR_4x
ACC_FACTOR='4x'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MODEL}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_4x

