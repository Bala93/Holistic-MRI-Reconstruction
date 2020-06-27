BATCH_SIZE=1
DEVICE='cuda:0'

EXP_NAME='varnet'
CHECKPOINT='/media/htic/NewVolume2/balamurali/fastmri_experiments/'${EXP_NAME}'/best_model.pt'
OUT_DIR='/media/htic/NewVolume2/balamurali/fastmri_experiments/'${EXP_NAME}'/results'
DATA_PATH='/media/htic/hd/fastmri/multicoil_test_v2/dataset'
DATA_CSV_PATH='/media/htic/NewVolume3/Balamurali/fastmri/multicoil_test.csv'
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --data-csv-path ${DATA_CSV_PATH} --challenge 'multicoil'
