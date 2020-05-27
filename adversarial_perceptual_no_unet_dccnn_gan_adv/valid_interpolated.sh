ALPHA=${1}
SAVE_MODEL='adversarial_perceptual_no_unet_wgan'
MODEL='deep_recon_synergy_net'
MODEL_1='adversarial_perceptual_no_unet_wgan'
DATASET_TYPE='cardiac'
BASE_PATH='/media/hticpose/drive2/Balamurali'
USMASK_PATH=${BASE_PATH}'/Reconstruction-for-MRI/us_masks/'${DATASET_TYPE}

<<ACC_FACTOR_2x
ACC_FACTOR='2x'
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
BATCH_SIZE=1
DEVICE='cuda:1'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
ACC_FACTOR_2x


#<<ACC_FACTOR_4x
ACC_FACTOR='4x'
CHECKPOINT=${BASE_PATH}'/experiments_reconseg/'${DATASET_TYPE}'/reconstruction/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
CHECKPOINT_1=${BASE_PATH}'/experiments_reconseg/'${DATASET_TYPE}'/reconstruction/acc_'${ACC_FACTOR}'/'${MODEL_1}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments_reconseg/'${DATASET_TYPE}'/reconstruction/acc_'${ACC_FACTOR}'/'${SAVE_MODEL}'/results_alpha_'${ALPHA}
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
python valid_interp.py --checkpoint_dccnn ${CHECKPOINT} --checkpoint_percept ${CHECKPOINT_1}  --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --interp_factor ${ALPHA} 
#ACC_FACTOR_4x

<<ACC_FACTOR_8x
ACC_FACTOR='8x'
CHECKPOINT=${BASE_PATH}'/experiments_reconseg/'${DATASET_TYPE}'/reconstruction/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments_reconseg/'${DATASET_TYPE}'/reconstruction/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
ACC_FACTOR_8x

