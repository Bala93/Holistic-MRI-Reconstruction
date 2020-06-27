from torch import nn
from models import NormUnet, NormRSN
import transforms as T
import torch



class SensitivityModel(nn.Module):
    def __init__(self, chans, num_pools):
        super().__init__()
        self.norm_unet = NormUnet(chans, num_pools)

    def chans_to_batch_dim(self, x):
        b, c, *other = x.shape
        return x.contiguous().view(b * c, 1, *other), b

    def batch_chans_to_chan_dim(self, x, batch_size):
        bc, one, *other = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, *other)

    def divide_root_sum_of_squares(self, x):
        return x / T.root_sum_of_squares_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(self, masked_kspace, mask):
        def get_low_frequency_lines(mask):
            l = r = mask.shape[-2] // 2
            while mask[..., r, :]:
                r += 1

            while mask[..., l, :]:
                l -= 1

            return l + 1, r

        l, r = get_low_frequency_lines(mask)
        num_low_freqs = r - l
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2
        x = T.mask_center(masked_kspace, pad, pad + num_low_freqs)
        x = T.ifft2(x)
        x, b = self.chans_to_batch_dim(x)
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)
        return x

class VarNetBlock(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.register_buffer('zero', torch.zeros(1, 1, 1, 1, 1))

    def forward(self, current_kspace, ref_kspace, mask, sens_maps):
        def sens_expand(x):
            return T.fft2(T.complex_mul(x, sens_maps))

        def sens_reduce(x):
            x = T.ifft2(x)
            return T.complex_mul(x, T.complex_conj(sens_maps)).sum(dim=1, keepdim=True)
    
        def soft_dc(x):
            return torch.where(mask, x - ref_kspace, self.zero) * self.dc_weight

        return current_kspace - \
                soft_dc(current_kspace) - \
                sens_expand(self.model(sens_reduce(current_kspace)))


class VariationalNetworkModel(nn.Module):
    
    def __init__(self, hparams):
        super().__init__()
        self.sens_net = SensitivityModel(hparams.sens_chans, hparams.sens_pools)
        self.cascades = nn.ModuleList([
            VarNetBlock(NormRSN(hparams.chans, hparams.pools))
            for _ in range(hparams.num_cascades)
        ])

    def forward(self, masked_kspace, mask):

        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        return T.root_sum_of_squares(T.complex_abs(T.ifft2(kspace_pred)), dim=1)
