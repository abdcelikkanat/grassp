import torch
from utils.common import EPS

# Constants for the computation of the imaginary erf function
_ERFI_P = .3275911
_ERFI_A1 = .254829592
_ERFI_A2 = -.284496736
_ERFI_A3 = 1.421413741
_ERFI_A4 = -1.453152027
_ERFI_A5 = 1.061405429


def remainder(x: torch.Tensor, y: float):
    """
    Compute the remainder of the division of x by y
    :param x: A tensor of shape (batch_size, ...)
    :param y: A float
    :return: A tensor of shape (batch_size, ...)
    """

    remainders = torch.remainder(x, y)
    remainders[torch.abs(remainders - y) < EPS] = 0

    return remainders


def div(x: torch.Tensor, y: float, decimals=6):
    """
    Compute the division of x by y
    :param x: A tensor of shape (batch_size, ...)
    :param y: A float
    :param decimals: The number of decimals to round to
    """

    return torch.round(torch.div(torch.round(x, decimals=decimals), y, ), decimals=decimals).type(torch.int)


def standardize(x: torch.Tensor):
    """
    Standardize a tensor of shape (B, N, D)/(B, D) to (B, N, D)/ (B, D)
    :param x: A tensor of shape (B, N, D)/(B, D)
    :return: A tensor of shape (B, N, D)/(B, D)
    """

    # If the input is None, return None
    if x is None:
        return x

    return x - torch.mean(x, dim=x.dim() - 2, keepdim=True)


def erfi_approx(z: torch.Tensor):
    """
    Approximate the erf function with maximum error of 1.5e-7,
    Source: Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables
            by Abramowitz M. and Stegun I.A. (Equation 7.1.26)
    :param z: The input value
    :return: The output value
    """
    z_ = 1j * z

    t = 1.0 / (1.0 + _ERFI_P * z_)

    return (1 - (t * (_ERFI_A1 + t * (_ERFI_A2 + t * (_ERFI_A3 + t * (_ERFI_A4 + t * _ERFI_A5))))) * torch.exp(
        -z_ ** 2)).imag
