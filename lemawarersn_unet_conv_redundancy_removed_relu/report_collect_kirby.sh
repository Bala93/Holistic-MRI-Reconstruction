
#MODEL='dc-rsn-feat-usereconfile'
#MODEL='lemawarersn_nc1_pretr_normaldAUnet_assistOlnyReNetworkwithrsn_noDC'
#MODEL='lemawarersn_nc3_pretr_rsn_normaldAUnet_assistOlnyReNetworkwithrsn'
#MODEL='lemawarersn_nc1_pretr_LemAssisteddASimpleUnetAssist_finetuneReNetworkwithrsn'
#MODEL='lemawaredcrsn_nc3_pretr_LemAssisteddASimpleUnetAssist_finetuneReNetworkwithrsn'
MODEL='dc-rsn-feat-redundancyremoved_UseAndAssistLatentDecoder'

BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
DATASET_TYPE='kirby90'
MASK_TYPE='cartesian'

echo ${MODEL}
echo ${DATASET_TYPE}
echo ${MASK_TYPE}


<<ACC_FACTOR_4x
ACC_FACTOR='4x'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/report.txt'
echo ${REPORT_PATH}
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
ACC_FACTOR_4x


#<<ACC_FACTOR_5x
ACC_FACTOR='5x'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/report.txt'
echo ${REPORT_PATH}
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
#ACC_FACTOR_5x


<<ACC_FACTOR_8x
ACC_FACTOR='8x'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/rsnlossfunctions/'${MODEL}'/report.txt'
echo ${REPORT_PATH}
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
ACC_FACTOR_8x



