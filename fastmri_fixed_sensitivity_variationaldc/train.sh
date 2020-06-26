DATAPATH=''
SAMPLERATE=''
DEVICE=''
EXPDIR=''
EXPNAME=''
NUMCASCADES=''
python train.py --resolution 320 --challenge multicoil --data-path ${DATAPATH} --sample-rate ${SAMPLERATE} --mode train --num-epochs 50 --device ${DEVICE} --exp-dir ${EXPDIR} --exp ${EXPNAME} --num-cascades ${NUMCASCADES} 