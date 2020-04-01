base_path='/data/balamurali'
acc_factor='5x'
dataset_type='kirby90'
mask_type='gaussian'
model_type='zf'
validation_path=${base_path}'/datasets/'${dataset_type}'/'${mask_type}'/validation/acc_'${acc_factor}
zf_save_path=${base_path}'/experiments/'${dataset_type}'/'${mask_type}'/acc_'${acc_factor}'/'${model_type}
echo python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} 
python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} 


