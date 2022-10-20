from typing import Tuple, List, Optional, Union, Dict, Any
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from piq.utils import _validate_input, _reduce
from piq.functional import gaussian_filter

def ssim(x: torch.Tensor, y: torch.Tensor, mask, kernel_size: int = 11, kernel_sigma: float = 1.5,
         data_range: Union[int, float] = 1., reduction: str = 'mean',
         downsample: bool = True, k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
    _validate_input([x, y], dim_range=(4, 5), data_range=(0, data_range))

    x = x / float(data_range)
    y = y / float(data_range)

    # Averagepool image if the size is large enough
    # f = max(1, round(min(x.size()[-2:]) / 256))
    # if (f > 1) and downsample:
    #     x = F.avg_pool2d(x, kernel_size=f)
    #     y = F.avg_pool2d(y, kernel_size=f)
    #     mask = F.avg_pool2d(mask, kernel_size=f)

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    ssim_map = _ssim_per_channel(x=x, y=y, mask=mask, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
    ssim_val = ssim_map.mean(1)

    ssim_val = _reduce(ssim_val, reduction)
    return ssim_val


def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, mask, kernel: torch.Tensor,
                      data_range: Union[float, int] = 1., k1: float = 0.01,
                      k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. '
                         f'Input size: {x.size()}. Kernel size: {kernel.size()}')

    c1 = k1 ** 2
    c2 = k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    if mask is not None:
        mask = F.conv2d(mask, weight=kernel, stride=1, padding=0, groups=n_channels)

    # Contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2. * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity (SSIM)
    ss = (2. * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    if mask is not None:
        ss = (ss * mask) / mask.mean()

    ssim_val = ss.mean(dim=(-1, -2))
    return ssim_val


def _reduce(x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    if reduction == 'none':
        return x
    elif reduction == 'mean':
        return x.mean(dim=0)
    elif reduction == 'sum':
        return x.sum(dim=0)
    else:
        raise ValueError("Unknown reduction. Expected one of {'none', 'mean', 'sum'}")


def _validate_input(
        tensors: List[torch.Tensor],
        dim_range: Tuple[int, int] = (0, -1),
        data_range: Tuple[float, float] = (0., -1.),
        # size_dim_range: Tuple[float, float] = (0., -1.),
        size_range: Optional[Tuple[int, int]] = None,
) -> None:
    if not __debug__:
        return

    x = tensors[0]

    for t in tensors:
        assert torch.is_tensor(t), f'Expected torch.Tensor, got {type(t)}'
        assert t.device == x.device, f'Expected tensors to be on {x.device}, got {t.device}'

        if size_range is None:
            assert t.size() == x.size(), f'Expected tensors with same size, got {t.size()} and {x.size()}'
        else:
            assert t.size()[size_range[0]: size_range[1]] == x.size()[size_range[0]: size_range[1]], \
                f'Expected tensors with same size at given dimensions, got {t.size()} and {x.size()}'

        if dim_range[0] == dim_range[1]:
            assert t.dim() == dim_range[0], f'Expected number of dimensions to be {dim_range[0]}, got {t.dim()}'
        elif dim_range[0] < dim_range[1]:
            assert dim_range[0] <= t.dim() <= dim_range[1], \
                f'Expected number of dimensions to be between {dim_range[0]} and {dim_range[1]}, got {t.dim()}'

        if data_range[0] < data_range[1]:
            assert data_range[0] <= t.min(), \
                f'Expected values to be greater or equal to {data_range[0]}, got {t.min()}'
            assert t.max() <= data_range[1], \
                f'Expected values to be lower or equal to {data_range[1]}, got {t.max()}'
