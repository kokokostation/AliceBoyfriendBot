from cityhash import CityHash64
import re
from collections import Counter
import pandas as pd
import os
import json
from copy import deepcopy

from batch_generator.dir import DirIterator
from train_utils.parallel_launcher import parallel_launcher
from dataset.reddit import is_unique
from train_utils.utils import get_files


CNTR_LOC = 'cntr.pickle'
REPLIES_DIR = 'replies'


def make_hash_of_entry(entry):
    return make_hash(entry[-1]['processed_body'])


def make_hash(body):
    body = re.sub('[^a-z]', '', body.lower())

    if body == '':
        return None

    return CityHash64(body)


def hash_gen(iterator, unique=False):
    for entry in iterator:
        hsh = make_hash_of_entry(entry)

        if hsh is not None:
            if unique:
                yield hsh, is_unique(entry)
            else:
                yield hsh


def make_priors_worker(files, data, index):
    train_iterator = DirIterator(files)

    cntr = Counter(hash_gen(train_iterator))

    return cntr


def make_top_replies_worker(files, data, index):
    hash_set, output_dir = data

    entries = []

    for entry in DirIterator(files):
        hsh = make_hash_of_entry(entry)

        if hsh in hash_set:
            entries.append((hsh, entry[-1]))

    with open(os.path.join(output_dir, str(index)), 'w') as outfile:
        json.dump(entries, outfile)


def queue_collector(hash_set, output_dir):
    hash_set = deepcopy(hash_set)
    entries = []

    for file in get_files(output_dir):
        with open(file, 'r') as infile:
            for hsh, entry in json.load(infile):
                if hsh in hash_set:
                    entries.append(entry)
                    hash_set.remove(hsh)

        os.remove(file)

    with open(os.path.join(output_dir, REPLIES_DIR), 'w') as outfile:
        json.dump(entries, outfile)


def get_hash_set(cntr, limit):
    return set([a for a, _ in cntr.most_common(limit)])


def make_priors(train_data, output_data, limit=2000000, files_num=10):
    output = parallel_launcher(train_data, None, make_priors_worker, 10, files_num)

    cntr = sum(output, Counter())

    pd.to_pickle(cntr, os.path.join(output_data, CNTR_LOC))

    hash_set = get_hash_set(cntr, limit)

    reply_data = os.path.join(output_data, REPLIES_DIR)
    os.makedirs(reply_data, exist_ok=True)

    parallel_launcher(train_data, (hash_set, reply_data),
                      make_top_replies_worker, 10, files_num)

    queue_collector(hash_set, reply_data)


class PriorGetter:
    def __init__(self, mc):
        self.mc = mc
        self.cntr = None

    def __call__(self, body):
        if self.cntr is None:
            self.cntr = self.mc.read('counter')

        return self.cntr[make_hash(body)]
