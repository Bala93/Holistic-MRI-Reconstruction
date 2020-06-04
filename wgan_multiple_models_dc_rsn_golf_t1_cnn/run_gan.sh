VAL_PATH='/media/hticpose/drive2/Balamurali/datasets/cardiac/validation/acc_4x' #validation path 
ACCELERATION='4' #acceleration factor
CHECKPOINT='/media/hticpose/drive2/Balamurali/experiments_reconglgan_setup/wgan_multiple_models_dcrsn/model_149.pt' #best_model.
OUT_DIR='/media/hticpose/drive2/Balamurali/experiments_reconglgan_setup/wgan_multiple_models_dcrsn/results' # Path to save reconstruction files 
python run.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --val-path ${VAL_PATH}
