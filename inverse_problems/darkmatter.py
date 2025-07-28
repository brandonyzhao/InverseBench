from inverse_problems.base import BaseOperator

import numpy as np
import torch
import torch.nn.functional as F
import string
import pdb

def tukey_lowpass(F, edge_frac=0.1):
    """
    Smooth low‑pass window for an n‑D complex spectrum `F`.
    
    Parameters
    ----------
    F : (*grid_shape, C) complex tensor
        Output of torch.fft.fftn on the spatial axes; channel dim last.
    edge_frac : float in (0,1)
        Width of the cosine taper expressed as a fraction of the
        Nyquist band (½ cycle/pixel).  edge_frac = 0 → brick‑wall,
        edge_frac = 0.1 → 10 % of the band rolled off smoothly.

        A good rule of thumb is    edge_frac = 1 / (2 * upsample_factor)
        so a 2× super‑sample uses a 0.25 roll‑off, 4× uses 0.125, etc.
        Decrease for a sharper (but more ringing‑prone) filter.
    """
    ndim = F.dim() - 1            # number of spatial dimensions
    device = F.device
    dtype  = F.dtype
    win_nd = 1.0                  # start with scalar 1

    for axis, N in enumerate(F.shape[:ndim]):
        # |f| ranges 0 … 0.5 in cycles / (original pixel)
        f = torch.fft.fftfreq(N, device=device)  # [-.5 … .5)
        f = f.abs()                              # symmetry
        f_nyq = 0.5
        f_pass = f_nyq * (1.0 - edge_frac)       # flat pass‑band edge

        # 1‑D raised‑cosine
        w = torch.ones_like(f, dtype=dtype)
        mask = (f > f_pass) & (f < f_nyq)
        w[mask] = 0.5 * (1 + torch.cos(
            torch.pi * (f[mask] - f_pass) / (f_nyq - f_pass)))
        w[f >= f_nyq] = 0.0                      # stop‑band

        # reshape so it can be broadcast along other axes & channels
        shape = [1] * (ndim + 1)                # +1 for channels
        shape[axis] = -1
        win_nd = win_nd * w.reshape(shape)

    return F * win_nd

def fourier_interp_torch(grid, coordinates, n=None):
  if not isinstance(grid, torch.Tensor):
      grid = torch.from_numpy(grid)
  if not isinstance(coordinates, torch.Tensor):
      coordinates = torch.from_numpy(coordinates)
  device = grid.device
  grid = grid.to(torch.complex64)
  coordinates = coordinates.to(torch.float32)
  ndim = grid.dim() - 1
  grid_shape, num_channels = grid.shape[:-1], grid.shape[-1]
  assert coordinates.shape[0] == ndim
  coord_shape = coordinates.shape[1:]
  coordinates = coordinates.reshape(ndim, -1)
  F = torch.fft.fftn(grid, dim=tuple(range(ndim)))
  if n is not None: 
    F = tukey_lowpass(F, edge_frac=1.0 / (2 * n))
  phases = []
  for c_i, s_i in zip(coordinates, grid_shape):
    freq = 2 * torch.pi * torch.fft.fftfreq(s_i, device=device)
    phases.append(torch.exp(1j * freq[:, None] * c_i[None, :]))
  grid_idx = string.ascii_uppercase[:ndim]
  fft_idx = grid_idx + 'c'
  phase_idx_list = [grid_idx[i] + 'n' for i in range(ndim)]
  einsum_str = fft_idx + ',' + ','.join(phase_idx_list) + '->nc'
  output_flat = torch.einsum(einsum_str, F, *phases).real
  output = output_flat.reshape(coord_shape + (num_channels,))
  norm_factor = torch.prod(torch.tensor(grid_shape, device=device)).item()
  output /= norm_factor
  return output

def ferp_planes_to_lightcone_single(density_plane, r, dx, coords, device): 
   im_shape = coords.size()[1:]

   plane_res = density_plane.size(0)
   od = density_plane

   foo = coords * r / dx - 0.5
   coords_in = foo + ((plane_res - 1 - foo.max()) / 2)

   g1, g2 = ks93inv(od, torch.zeros_like(od), device)
   
   # Note: I am doing the shear convolution before the interpolation due to the width of the kernel in fourier space
   # Interpolate at the density plane coordinates
   
   coords_fourier = coords_in.reshape(2, -1)
   im = fourier_interp_torch(od[..., None], coords_fourier).reshape(im_shape)
   g1_im = fourier_interp_torch(g1[..., None], coords_fourier).reshape(im_shape)
   g2_im = fourier_interp_torch(g2[..., None], coords_fourier).reshape(im_shape)
    
   return im, g1_im, g2_im

ferp_planes_to_lightcone_vmap = torch.vmap(ferp_planes_to_lightcone_single, in_dims=(0, 0, 0, None, None), out_dims=-1)

def lerp_planes_to_lightcone(dm_cube, r, dx, coords, device):
    num_planes = dm_cube.size(0)
    plane_res = dm_cube.size(1)

    coords_flip = torch.flip(torch.permute(coords, (1, 2, 0)), (-1,)).unsqueeze(0)
    foo = coords_flip * r.reshape(num_planes, 1, 1, 1) / dx.reshape(num_planes, 1, 1, 1)
    coords_grid = (2 * foo - 1. - foo.amax(dim=(1, 2, 3), keepdim=True)) / plane_res

    g1, g2 = ks93inv_batch(dm_cube, torch.zeros_like(dm_cube), device)
    grid_stack = torch.stack([dm_cube, g1, g2], dim=1)

    patch = torch.nn.functional.grid_sample(
        grid_stack,
        coords_grid,
        mode='bilinear', 
        padding_mode='zeros',
        align_corners=False
    )

    return patch

def ks93inv(kE, kB, device):
    # Check consistency of input maps
    assert kE.shape == kB.shape

    # Compute Fourier space grids
    (nx, ny) = kE.shape
    k1, k2 = torch.meshgrid(torch.fft.fftfreq(ny), torch.fft.fftfreq(nx), indexing='ij')
    k1 = k1.to(device)
    k2 = k2.to(device)

    # Compute Fourier transforms of kE and kB
    kEhat = torch.fft.fft2(kE)
    kBhat = torch.fft.fft2(kB)

    # Apply Fourier space inversion operator
    p1 = k1 * k1 - k2 * k2
    p2 = 2 * k1 * k2
    k2 = k1 * k1 + k2 * k2
    k2[0, 0] = 1. #avoid division by 0
    g1hat = (p1 * kEhat - p2 * kBhat) / k2
    g2hat = (p2 * kEhat + p1 * kBhat) / k2

    # Transform back to pixel space
    g1 = torch.fft.ifft2(g1hat).real
    g2 = torch.fft.ifft2(g2hat).real

    return g1, g2

ks93inv_batch = torch.vmap(ks93inv, in_dims=(0, 0, None), out_dims=(0, 0))

def weighted_proj(vol3d, pws): 
    foo = vol3d.reshape(pws.shape[0], pws.shape[1], vol3d.shape[-1])
    vol_weighted  = foo * pws 
    return vol_weighted.sum(1)

weighted_proj_map = torch.vmap(weighted_proj, in_dims=(None, 0), out_dims=(0,))

class DarkMatterImaging(BaseOperator):

    '''
    Dark matter imaging forward model
    '''

    def __init__(self, gamma, num_iters, pws, eta_shape, lightcone_specs, coords, do_proj, 
                 sigma_noise, unnorm_shift, unnorm_scale, device) -> None:
        super().__init__(sigma_noise, unnorm_shift, unnorm_scale, device)
        self.gamma = gamma
        self.num_iters = num_iters
        self.pws = torch.tensor(np.load(pws)).to(device)
        self.eta_shape = tuple(eta_shape)
        self.od_bm = torch.zeros(self.eta_shape).to(device)
        self.lightcone_specs = torch.load(lightcone_specs) 
        self.r = self.lightcone_specs['r'].to(device)
        self.dx = self.lightcone_specs['dx'].to(device)
        self.coords = torch.tensor(np.load(coords)).float().to(device)
        self.do_proj = do_proj
        self.pad_mat = F.pad(torch.ones((120, 120)), (4,4,4,4), 'constant', 0).unsqueeze(0).to(device)

        self.class_labels = torch.eye(self.eta_shape[0]).to(device) # class label matrix for sampling cubes

    def forward(self, x, **kwargs):
        x = self.unnormalize(x).reshape(self.eta_shape)
        # Try padding 
        # x = x * self.pad_mat
        # pdb.set_trace()
        lightcone_out = lerp_planes_to_lightcone(x, self.r, self.dx, self.coords, self.device)
        k = lightcone_out[:, 0]
        e1 = lightcone_out[:, 1]
        e2 = lightcone_out[:, 2]
    
        output = lightcone_out[1:, 1:]

        # if self.do_proj:
        #     output =  weighted_proj_map(output, self.pws)
        return output#.reshape(20, -1)
    
    def ferp_lightcone(self, x): 
        x = self.unnormalize(x)
        x = x.reshape(self.eta_shape)
        k, _, _ = ferp_planes_to_lightcone_vmap(x, self.r, self.dx, self.coords, self.device)
        return k
    
    def lerp_lightcone(self, x): 
        x = self.unnormalize(x)
        x = x.reshape(self.eta_shape)
        grid = lerp_planes_to_lightcone(x, self.r, self.dx, self.coords, self.device)
        return grid[:, 0]
    
    @staticmethod
    def mean_subtract(x): 
        return x - torch.mean(x, dim=(1,2), keepdim=True)
    
    def loss(self, pred, observation, **kwargs): 
        # MSE Loss
        # pred = self.mean_subtract(pred.reshape(self.eta_shape))
        # observation = self.mean_subtract(observation.reshape(self.eta_shape))
        return torch.mean(torch.square(self.forward(pred) - observation)) 

    def loss_m(self, measurements, observation): 
        # Chi-sq observation loss 
        return torch.mean(torch.square(measurements - observation))
    
    def evaluate_chisq(self, pred, observation): 
        return (torch.mean(torch.square(self.forward(pred) - observation)) / self.sigma_noise).cpu().item()
    
    def evaluate_mse(self, target, pred): 
        target = target.reshape(self.eta_shape)
        pred = pred.reshape(self.eta_shape)
        return torch.mean(torch.square(self.mean_subtract(target) - self.mean_subtract(pred))).cpu().item()
    
    def evaluate_psnr(self, target, pred, max_val=1.0): 
        target = target.reshape(self.eta_shape)
        pred = pred.reshape(self.eta_shape)
        mse = torch.mean(torch.square(self.mean_subtract(target) - self.mean_subtract(pred)))
        psnr = 10 * torch.log10(max_val**2 / (mse + 1e-8))  # epsilon to avoid log(0)
        return psnr.cpu().item()
    
    def evaluate_pcc(self, target, pred): 
        target = target.reshape(self.eta_shape)
        pred = pred.reshape(self.eta_shape)

        target = self.mean_subtract(target)
        pred = self.mean_subtract(pred)

        # Flatten each map to 1D
        target_flat = target.view(target.size(0), -1)
        pred_flat = pred.view(pred.size(0), -1)

        # Compute PCC per map in the batch
        numerator = torch.sum(target_flat * pred_flat, dim=1)
        denominator = (
            torch.norm(target_flat, dim=1) * torch.norm(pred_flat, dim=1) + 1e-8  # avoid division by zero
        )
        pcc_per_map = numerator / denominator  # shape: [20]

        return pcc_per_map.mean().cpu().item()  # return average PCC over the batch