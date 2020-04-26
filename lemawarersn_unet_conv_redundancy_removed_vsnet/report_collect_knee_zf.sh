MODEL='zf'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='knee-mc-axial-t2'   #knee-mc-coronal-pd,knee-mc-coronal-pd-fs

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

