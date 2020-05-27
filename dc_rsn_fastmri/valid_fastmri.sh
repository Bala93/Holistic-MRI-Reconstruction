BATCH_SIZE=1
DEVICE='cuda:1'

EXP_NAME='dc_rsn'
CHECKPOINT='/media/htic/NewVolume2/balamurali/fastmri_experiments/'${EXP_NAME}'/best_model.pt'
OUT_DIR='/media/htic/NewVolume2/balamurali/fastmri_experiments/'${EXP_NAME}'/results'
DATA_PATH='/media/htic/NewVolume5/fastmri/singlecoil_test_v2'
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} 

