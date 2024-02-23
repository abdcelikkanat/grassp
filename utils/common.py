import os
import math
import random
import torch

# Path definitions
BASE_FOLDER = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

# Constants
EPS = 1e-6
INF = 1e+6
PI = math.pi
LOG2PI = math.log(2*PI)


def set_seed(seed):
    """
    Set the random seed for all the random number generators
    :param seed: the seed value
    """

    random.seed(seed)
    torch.manual_seed(seed)


def pair_iter(n, is_directed=False):
    """
    A method generating all the pairs of nodes
    :param n: the number of nodes
    :param is_directed: whether the graph is directed or not
    """

    if is_directed:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                else:
                    yield i, j

    else:
        for i in range(n):
            for j in range(i+1, n):
                yield i, j


def matIdx2flatIdx(i, j, n, is_directed: bool = False, dtype: torch.dtype = torch.long):
    """
    A method converting mat index to flat index
    :param i: the first index
    :param j: the second index
    :param n: the number of nodes
    :param is_directed: whether the graph is directed or not
    """

    if not is_directed:

        return ( (n-1) * i - (i*(i+1)//2) + (j-1) ).to(dtype)

    else:

        return ( i*n + j - i - 1*(j > i) ).to(dtype)


def flatIdx2matIdx(idx, n, is_directed=False, dtype: torch.dtype = torch.long):
    """
    A method converting flat index to mat index
    :param idx: the flat index
    :param n: the number of nodes
    :param dtype: the data type
    :param is_directed: whether the graph is directed or not
    """

    if is_directed:

        row_idx = idx // (n-1)
        col_idx = idx % (n-1)
        col_idx[col_idx >= row_idx] += 1

        return torch.vstack((row_idx, col_idx)).to(dtype)

    else:
        # Because of the numerical issues, as a temporary solution, we use the following code
        if n > 3000:

            rc = torch.index_select(torch.triu_indices(n, n, 1, device=idx.device), dim=1, index=idx)
            r, c = rc[0], rc[1]

        else:

            r = torch.ceil(n - 1 - 0.5 - torch.sqrt(n ** 2 - n - 2 * idx - 1.75)).type(dtype)
            c = idx - r * n + ((r + 1) * (r + 2)) // 2

        return torch.vstack((r.type(dtype), c.type(dtype)))

