<<setup
MODEL='zf'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='knee-mc-axial-t2'
setup

<<ACC_FACTOR_4x
ACC_FACTOR='4x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/'
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MODEL}'/results/'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MODEL}'/'
echo python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
ACC_FACTOR_4x


TARGET_PATH='/media/htic/NewVolume5/knee_mri_vsnet_globus/axial_t2_h5/validation'
PREDICTIONS_PATH='/media/htic/NewVolume5/knee_mri_vsnet_globus/axial_t2_h5/zf/results'
REPORT_PATH='/media/htic/NewVolume5/knee_mri_vsnet_globus/axial_t2_h5/zf'
echo python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
sudo /home/htic/anaconda2/envs/torch4/bin/python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
