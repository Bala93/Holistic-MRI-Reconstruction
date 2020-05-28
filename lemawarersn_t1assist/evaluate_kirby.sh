#MODEL='dc-rsn-feat-usereconfile'
#MODEL='lemawarersn_nc1_pretr_normaldAUnet_assistOlnyReNetworkwithrsn_noDC'
#MODEL='lemawarersn_nc3_pretr_rsn_normaldAUnet_assistOlnyReNetworkwithrsn'
#MODEL='lemawarersn_nc1_pretr_LemAssisteddASimpleUnetAssist_finetuneReNetworkwithrsn'
#MODEL='lemawaredcrsn_nc3_pretr_LemAssisteddASimpleUnetAssist_finetuneReNetworkwithrsn'
MODEL='dc-rsn-feat-redundancyremoved_UseAndAssistLatentDecoder'
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='kirby90'
MASK_TYPE='cartesian'


<<ACC_FACTOR_4x
ACC_FACTOR='4x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/results'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/'
echo python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
ACC_FACTOR_4x

#<<ACC_FACTOR_5x
ACC_FACTOR='5x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/results'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/'
echo python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
#ACC_FACTOR_5x




