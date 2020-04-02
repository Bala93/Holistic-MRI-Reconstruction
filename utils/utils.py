import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.fft import fft, fft2, ifft2, ifft, ifftshift, fftshift
import random
import shutil as sh

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask(shape, acc, sample_n=10, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = ifftshift(mask, axes=(-1, -2))

    return mask

def normalize_volume(img_vol):

    h,w = img_vol.shape[0],img_vol.shape[1]
    norm_img_vol = np.empty([h,w,0])

    for sl_no in range(img_vol.shape[-1]):

        img = img_vol[:,:,sl_no]
        img_norm = img / float(np.max(img))

        norm_img_vol = np.dstack([norm_img_vol,img_norm])

    return norm_img_vol

def undersample_volume_static(img_vol,us_factors,us_masks):
    
    h,w = img_vol.shape[0],img_vol.shape[1]

    us_volumes = {}

    for ii in range(len(us_factors)):
        print("us_factor: ",us_factors[ii])
        us_factor = us_factors[ii]
        us_mask   = us_masks[ii]

        us_img_vol    = np.empty([h,w,0])
        us_kspace_vol = np.empty([h,w,0])

        for sl_no in range(img_vol.shape[-1]):
            #print("sl_no: ",sl_no)

            img = img_vol[:,:,sl_no]

            kspace     = np.fft.fft2(img,norm='ortho') 
            us_kspace  = kspace * us_mask 
            us_kspace_vol = np.dstack([us_kspace_vol,us_kspace])

            us_img        = np.abs(np.fft.ifft2(us_kspace,norm='ortho'))
            us_img_vol    = np.dstack([us_img_vol,us_img])    

        us_volumes['img_volus_{}x'.format(str(us_factor))] = us_img_vol
        us_volumes['kspace_volus_{}x'.format(str(us_factor))] = us_kspace_vol
            
    return us_volumes


def undersample_volume_dynamic(img_vol,us_factors):
    
    h,w = img_vol.shape[0],img_vol.shape[1]

    us_volumes = {}

    for us_factor in us_factors:
        us_img_vol  = np.empty([h,w,0])

        for sl_no in range(img_vol.shape[-1]):

            img = img_vol[:,:,sl_no]
            img_us   = np.fft.ifft2(np.fft.fft2(img) * cartesian_mask((h,w),us_factor)).real
            img_us  /= float(np.max(img_us))
        
            us_img_vol =  np.dstack([us_img_vol,img_us])    

        us_volumes['volus_{}x'.format(str(us_factor))] = us_img_vol
            
    return us_volumes


def create_dataset_split(src_dir,train_dir,validation_dir,file_ext,split_ratio=0.7):

    file_list = glob.glob(os.path.join(src_dir,'.*{}'.format(file_ext))) #Works only for files in the first level of the folder structure 
    random.shuffle(file_list)

    train_file_count = int(len(file_list) * split_ratio)

    train_files = file_list[:train_file_count]
    validation_files = file_list[train_file_count:]

    for file_path in train_files:
        dst_path = os.path.join(train_dir,os.path.basename(file_path))
        sh.copy(file_path,dst_path)

    for file_path in validation_files:
        dst_path = os.path.join(validation_dir,os.path.basename(file_path))
        sh.copy(file_path,dst_path)
        
    return 

def range_np(input):

    return np.max(input),np.min(input)



