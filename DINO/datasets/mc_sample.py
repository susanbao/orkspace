import os
import math
import numpy as np
import torch
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
from torch.utils.data import Sampler

class MCSampler(Sampler[int]):
    r"""Random select M data and samples them sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized
    
    def __init__(self, data_source: Sized, sample_len: int) -> None:
        self.data_source = data_source
        self.sample_len = sample_len
        if len(self.data_source) < self.sample_len:
            raise ValueError("Error on MCSampler")

    def __iter__(self) -> Iterator[int]:
        perm = torch.randperm(len(self.data_source))
        self.iter_list = perm[:self.sample_len]
        return iter(self.iter_list)

    def __len__(self) -> int:
        return self.sample_len