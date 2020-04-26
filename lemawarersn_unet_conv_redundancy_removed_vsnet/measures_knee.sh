MODEL='vs-net-recon-synergy-net'
#DATASET_TYPE='knee-mc-coronal-pd'
#DATASET_TYPE='knee-mc-coronal-pd-fs'
DATASET_TYPE='knee-mc-axial-t2'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
#MASK_TYPE='gaussian'
#MASK_TYPE='cartesian'


ACC_FACTOR='5x'
#TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/'
TARGET_PATH='/media/htic/NewVolume5/knee_mri_vsnet_globus/axial_t2_h5/validation'
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}
echo ${TARGET_PATH}
echo ${PREDICTIONS_PATH}
echo ${REPORT_PATH}
python measures_csv.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH}



<<ACC_FACTOR_8x
ACC_FACTOR='8x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
python measures_csv.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH}
#ACC_FACTOR_8x
