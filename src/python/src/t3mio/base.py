# -*- coding: utf-8 -*-            
# @Time : 2025/1/16 14:42
# @Autor: joanzhong
# @FileName: base.py
# @Software: IntelliJ IDEA

from typing import List, Callable, Optional
import os
import random
from torch.utils.data import IterableDataset, get_worker_info


def default_map_func(x):
    return x


class FiniteStreamDataset(IterableDataset):
    """Iterable dataset that support the `map` and `shuffle` operator.
    The total number of samples and the number of samples on each node must be known.
    """
    def __init__(self, slice_id: Optional[int] = None, slice_count: Optional[int] = None) -> None:
        """
        Args:
            slice_id: ID (rank) of current worker.
            slice_count: Number of workers (world size) total.
        """
        super().__init__()

        if slice_id is None:
            slice_id = int(os.environ.get('RANK', 0))
        if slice_count is None:
            slice_count = int(os.environ.get('WORLD_SIZE', 1))
        self._seed = 0
        self._epoch = 0
        self._slice_id = slice_id
        self._slice_count = slice_count
        self._map_func: Callable = default_map_func
        self._buffer_size: int = 1

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    @property
    def do_shuffle(self) -> bool:
        return self._buffer_size == 1

    @property
    def slice_id(self) -> int:
        return self._slice_id

    @property
    def slice_count(self) -> int:
        return self._slice_count

    def __len__(self):
        return self.num_samples_local

    @property
    def num_samples_global(self) -> int:
        """Return the number of samples."""
        raise NotImplementedError("Please implement this method in subclass")

    @property
    def num_samples_local(self) -> int:
        """Return the number of this rank."""
        raise NotImplementedError("Please implement this method in subclass")

    def map(self, func: Callable):
        """A function that map a raw input line to a sample."""
        self._map_func = func
        return self

    def shuffle(self, buffer_size: int):
        """Shuffle the iterable dataset with a buffer size `buffer_size`."""
        assert buffer_size >= 1
        self._buffer_size = buffer_size
        return self

    def build_data_stream(self, worker_id: int, num_workers: int):
        raise NotImplementedError("Please implement this method in subclass")

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        def iterator():
            yield from self.build_data_stream(worker_id, num_workers)

        def buffer_iterator():
            buf: List = []
            rnd_generator = random.Random()
            rnd_generator.seed(self._seed + self._epoch)
            for x in self.build_data_stream(worker_id, num_workers):
                if len(buf) == self._buffer_size:
                    idx = rnd_generator.randint(0, self._buffer_size - 1)
                    yield buf[idx]
                    buf[idx] = x
                else:
                    buf.append(x)
            rnd_generator.shuffle(buf)
            while buf:
                yield buf.pop()

        return iterator() if self._buffer_size == 1 else buffer_iterator()