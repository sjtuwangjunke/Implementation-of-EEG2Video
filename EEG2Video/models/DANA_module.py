import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import math
import os.path as osp

class Diffusion(object):
    def __init__(self, time_steps):
        self.time_steps = time_steps
        self.betas = self._linear_beta_schedule()

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # coeff1
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def _get_index_from_list(self, vals: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """ helper function to get index from list, considering batch dimension

        Args:
            vals (torch.Tensor): list of values
            t (torch.Tensor): timestep
            x_shape (torch.Size): shape of input image

        Returns:
            torch.Tensor: value at timestep t
        """
        batch_size = t.shape[0]  # batch_size
        out = vals.gather(-1, t.cpu())  # (batch_size, 1)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def _linear_beta_schedule(self, start=0.0001, end=0.02) -> torch.Tensor:
        """ linear beta schedule
        Args:
            start (float, optional): beta at timestep 0. Defaults to 0.0001.
            end (float, optional): beta at last timestep. Defaults to 0.02.

        Returns:
            torch.Tensor: beta schedule
        """
        return torch.linspace(start, end, self.time_steps)

    def forward(self, x_0, dynamic_beta):
        b, f, c, h, w = x_0.shape
        t = torch.randint(0, self.time_steps, (b,)).cuda().long()

        diverse_noise = torch.randn_like(x_0).to(self.device)
        same_noise_i = torch.randn((b, 1, c, h, w)).to(self.device)
        same_noise = same_noise_i.repeat(1, f, 1, 1, 1)

        diverse_noise = diverse_noise * math.sqrt(1-dynamic_beta)
        same_noise = same_noise * math.sqrt(dynamic_beta)

        sqrt_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_alphas_cumprod, t, x_0.shape
        )

        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) \
               + sqrt_one_minus_alphas_cumprod_t.to(self.device) * (diverse_noise + same_noise).to(self.device)



if __name__ == '__main__':
    x_0 = np.load('latent_out_40_classes_test.npy')
    x_0 = torch.from_numpy(x_0)
    print(x_0.shape)
    dynamic_beta = 0.3

    x = torch.randn((2,3,4))
    print(x.shape)
    print(x)
    delta = [0.2, 1]
    for i in range(x.shape[0]):
        x[i] = x[i] * delta[i]
    print(x)
    #
    # diffusion = Diffusion(time_steps=500)
    #
    # out = diffusion.forward(x_0, dynamic_beta)
    # print(out.shape)
