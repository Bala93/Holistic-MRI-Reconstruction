import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from runstats import Statistics
from skimage.measure import compare_psnr, compare_ssim
from skimage.filters import laplace
from tqdm import tqdm
from sewar.full_ref import vifp,_vifp_single

# adding hfn metric 
def hfn(gt,pred):

    hfn_total = []

    for ii in range(gt.shape[-1]):
        gt_slice = gt[:,:,ii]
        pred_slice = pred[:,:,ii]

        pred_slice[pred_slice<0] = 0 #bring the range to 0 and 1.
        pred_slice[pred_slice>1] = 1

        gt_slice_laplace = laplace(gt_slice)        
        pred_slice_laplace = laplace(pred_slice)

        hfn_slice = np.sum((gt_slice_laplace - pred_slice_laplace) ** 2) / np.sum(gt_slice_laplace **2)
        hfn_total.append(hfn_slice)

    return np.mean(hfn_total)


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    #return compare_ssim(
    #    gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    #)
    return compare_ssim(gt,pred,multichannel=True, data_range=gt.max())

def vif_p(gt, pred):
    return vifp(gt,pred)

def vif_n(gt,pred):
    return _vifp_single(gt[:,:,0],pred[:,:,0],4);


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    HFN=hfn,
    VIF_P=vif_p,
    VIF=vif_n
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }


    '''
    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )
    '''

    def get_report(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )




def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)

    for tgt_file in args.target_path.iterdir():
        #print (tgt_file)
        with h5py.File(tgt_file) as target, h5py.File(
          args.predictions_path / tgt_file.name) as recons:
            target = np.expand_dims(target[recons_key].value , axis = 2)
            recons = recons['reconstruction'].value.squeeze(0)
            recons = np.transpose(recons,[1,2,0])
            #print (target.shape,recons.shape)
            #print(tgt_file)
            metrics.push(target, recons)
            
    return metrics


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--report-path', type=pathlib.Path, required=True,
                        help='Path to save metrics')
    parser.add_argument('--interp_factor', type=float, required=True,
                        help='Path to save metrics')

    args = parser.parse_args()

    recons_key = 'volfs'
    metrics = evaluate(args, recons_key)
    metrics_report = metrics.get_report()
    save_file_name = 'report_alpha_'+ str(args.interp_factor) +'.txt'
    with open(args.report_path / save_file_name, 'w') as f:
        f.write(metrics_report)

    #print(metrics)
