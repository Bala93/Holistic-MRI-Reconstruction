
from models import  UnetModel
import glob
import h5py
import os 

device = 'cuda:0'

model = UnetModel(1,1,32,4,0).to(device)
print (model)

recon_h5_path = '/media/htic/NewVolume1/murali/MR_reconstruction/experiments/kirby90/cartesian/acc_4x/ablative-dA-fastMRIUNet-wang_etal-three-channels/results/*.h5' # reconsturctions saved path (kirby 4x)

h5_files = glob.glob(recon_h5_path)

dst_dir  = '/media/htic/NewVolume1/murali/MR_reconstruction/experiments/kirby90/cartesian/acc_4x/dc-rsn-unet-lem-feature'

for h5_file in h5_files:

   fname = os.path.basename(h5_file)

   with h5py.File(h5_file,'r') as hf:
      
       recon = hf['reconstruction'].value

       print (recon.shape)

   with h5py.File(os.path.join(dst_dir,fname),'w') as hf:


       recon = recon.to(device)

       feat,_ = model(recon).cpu().numpy()

       print (feat.shape)

   break
       
