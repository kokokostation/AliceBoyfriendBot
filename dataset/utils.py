import os
import pandas as pd
from itertools import chain
import numpy as np
import re
import json

from train_utils.utils import write_json


def shuffle(data_dir, output_dir, limit=None, step=50000):
    pairs = chain.from_iterable(pd.read_json(os.path.join(data_dir, fname), typ='series')
                                for fname in os.listdir(data_dir))
    pairs = [item for item in pairs
             if all((limit is None or len(text) < limit) and
                    (re.match('^[^"\d]+$', text) is not None)
                    for _, text, _ in item)]

    np.random.shuffle(pairs)

    for num, i in enumerate(range(0, len(pairs), step)):
        write_json(pairs[i:i + step], os.path.join(output_dir, str(num)))


class PackMaker:
    def __init__(self, pack_size, output_dir, prefix, writer=write_json):
        self.file_no = 0
        self.pack_size = pack_size
        self.output_dir = output_dir
        self.prefix = prefix
        self.writer = writer

    def make_pack(self, portion, finalize=False):
        while len(portion) > self.pack_size or finalize:
            fname = os.path.join(self.output_dir, '{}_{}'.format(self.prefix, self.file_no))
            self.writer(portion[-self.pack_size:], fname)

            del portion[-self.pack_size:]
            self.file_no += 1
            finalize = False

    def finalize(self, portion):
        self.make_pack(portion, True)