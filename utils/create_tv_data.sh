base_path='/data/balamurali'
acc_factor='4x'
dataset_type='kirby90'
mask_type='cartesian'
model_type='tv'

validation_path=${base_path}'/datasets/'${dataset_type}'/'${mask_type}'/validation/acc_'${acc_factor}
tv_save_path=${base_path}'/experiments/'${dataset_type}'/'${mask_type}'/acc_'${acc_factor}'/'${model_type}'/results'

echo python val_tv_regularizer.py --validation_path ${validation_path} --tv_save_path ${tv_save_path} --acc_factor ${acc_factor} 
python val_tv_regularizer.py --validation_path ${validation_path} --tv_save_path ${tv_save_path} --acc_factor ${acc_factor} 
