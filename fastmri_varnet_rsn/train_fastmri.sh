TRAINPATH='/media/htic/NewVolume5/fastmri/multicoil_train/dataset'
VALIDATIONPATH='/media/htic/hd/fastmri/multicoil_valid/dataset'
TRAINCSVPATH='/media/htic/NewVolume3/Balamurali/fastmri/multicoil_train.csv'
VALIDCSVPATH='/media/htic/NewVolume3/Balamurali/fastmri/multicoil_valid.csv'
SAMPLERATE=0.1
DEVICE='cuda:0'
EXPNAME='varnet_rsn'
EXPDIR='/media/htic/NewVolume2/balamurali/fastmri_experiments/'${EXPNAME}
NUMCASCADES=2

python train.py --resolution 320 --challenge multicoil --train-path ${TRAINPATH} --validation-path ${VALIDATIONPATH} --train-csv-path ${TRAINCSVPATH} --validation-csv-path ${VALIDCSVPATH} --sample-rate ${SAMPLERATE} --mode train --num-epochs 50 --device ${DEVICE} --exp-dir ${EXPDIR} --num-cascades ${NUMCASCADES} 
