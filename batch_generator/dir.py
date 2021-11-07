import json
from copy import copy

import numpy as np

from batch_generator.utils import get_random_paths
from train_utils.utils import read_json


class DirIterator:
    def __init__(self, fnames, reader=read_json, shuffle=True):
        self.fnames = fnames
        self.reader = reader
        self.gen = None
        self.shuffle = shuffle

    @staticmethod
    def from_data_folder(data_folder, file_limit=None, reader=read_json, shuffle=True):
        return DirIterator(get_random_paths(data_folder, file_limit), reader, shuffle)

    def __iter__(self):
        self.gen = self.helper()

        return self

    def __next__(self):
        if self.gen is None:
            self.gen = self.helper()

        return next(self.gen)

    def helper(self):
        fnames = copy(self.fnames)

        if self.shuffle:
            np.random.shuffle(fnames)

        for fname in fnames:
            data = self.reader(fname)

            if self.shuffle:
                np.random.shuffle(data)

            for item in data:
                yield item


def dir_iterator(data_folder, file_limit, make_validation=False, val_size=1):
    if make_validation:
        fnames = get_random_paths(data_folder, file_limit)

        train_fnames = fnames[:-val_size]
        val_fnames = fnames[-val_size:]

        return DirIterator(train_fnames), \
               DirIterator(val_fnames)
    else:
        return DirIterator.from_data_folder(data_folder, file_limit)
