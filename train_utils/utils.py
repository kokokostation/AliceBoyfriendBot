import json
import dill
from itertools import chain
import os
from functools import partial


def write_json(obj, path):
    with open(path, 'w') as outfile:
        json.dump(obj, outfile)


def read_json(path):
    with open(path, 'r') as infile:
        return json.load(infile)


def flatten(placeholders, batch):
    if type(placeholders) in [list, tuple]:
        return chain.from_iterable([flatten(p, b) for p, b in zip(placeholders, batch)])
    else:
        return [(placeholders, batch)]


def make_feed_dict(placeholders, batch):
    return dict(flatten(placeholders, batch))


def read_dill(path):
    with open(path, 'rb') as infile:
        return dill.load(infile)


def write_dill(obj, path):
    with open(path, 'wb') as outfile:
        dill.dump(obj, outfile)


def get_files(data_dir, folders_ok=False):
    all_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]

    if not folders_ok:
        all_files = [file for file in all_files if os.path.isfile(file)]

    return all_files
