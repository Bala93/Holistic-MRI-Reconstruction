#<<2x
acc_factor='2x'
dataset_type='cardiac'
validation_path='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${dataset_type}'/validation/acc_'${acc_factor}
zf_save_path='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${dataset_type}'/zf_data/acc_'${acc_factor}
echo python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} 
python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} &
#2x


#<<2.5x
acc_factor='2_5x'
dataset_type='cardiac'
validation_path='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${dataset_type}'/validation/acc_'${acc_factor}
zf_save_path='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${dataset_type}'/zf_data/acc_'${acc_factor}
echo python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} 
python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} &
#2.5x


#<<3.3x
acc_factor='3_3x'
dataset_type='cardiac'
validation_path='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${dataset_type}'/validation/acc_'${acc_factor}
zf_save_path='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${dataset_type}'/zf_data/acc_'${acc_factor}
echo python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} 
python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} &
#3.3x

#<<4x
acc_factor='4x'
dataset_type='cardiac'
validation_path='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${dataset_type}'/validation/acc_'${acc_factor}
zf_save_path='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${dataset_type}'/zf_data/acc_'${acc_factor}
echo python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} 
python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} &
#4x

#<<5x
acc_factor='5x'
dataset_type='cardiac'
validation_path='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${dataset_type}'/validation/acc_'${acc_factor}
zf_save_path='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${dataset_type}'/zf_data/acc_'${acc_factor}
echo python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} 
python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} &
#5x

#<<8x
acc_factor='8x'
dataset_type='cardiac'
validation_path='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${dataset_type}'/validation/acc_'${acc_factor}
zf_save_path='/media/htic/NewVolume1/murali/MR_reconstruction/datasets/'${dataset_type}'/zf_data/acc_'${acc_factor}
echo python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} 
python create_zf_data.py --validation_path ${validation_path} --zf_save_path ${zf_save_path} --acc_factor ${acc_factor} --dataset_type ${dataset_type} 
#8x


