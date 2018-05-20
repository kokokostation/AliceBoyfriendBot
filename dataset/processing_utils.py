import os
import json
import numpy as np
from shutil import rmtree

from train_utils.parallel_launcher import parallel_launcher
from train_utils.utils import get_files
from dataset.utils import PackMaker


def shuffle_batch(files, step, index):
    buf = []

    for file in files:
        with open(file, 'r') as infile:
            buf.extend(json.load(infile))

    np.random.shuffle(buf)

    for i, file in enumerate(files):
        with open(file, 'w') as outfile:
            json.dump(buf[i * step:(i + 1) * step], outfile)


def filter_batch(files, data, index):
    filter_f, output_dir = data

    for file in files:
        with open(file, 'r') as infile:
            data = json.load(infile)

        with open(os.path.join(output_dir, os.path.basename(file)), 'w') as outfile:
            json.dump(list(filter(filter_f, data)), outfile)


def compress_batch(files, data, index):
    step, output_dir, buffer_dir = data

    portion = []
    pack_maker = PackMaker(step, output_dir, index)
    for file in files:
        with open(file, 'r') as infile:
            data = json.load(infile)

        portion.extend(data)

        os.remove(file)

        pack_maker.make_pack(portion)

    pack_maker.finalize(portion)


def rename(data_dir):
    for i, file in enumerate(get_files(data_dir)):
        os.rename(file, os.path.join(os.path.dirname(file), str(i)))


def filter_and_compress(data_dir, filter_f, output_dir):
    parallel_launcher(data_dir, (filter_f, output_dir), filter_batch, 10, 20)

    buffer_dir = os.path.join(output_dir, 'buffer')
    os.makedirs(buffer_dir)
    data = 100000, output_dir, buffer_dir

    result = parallel_launcher(output_dir, data, compress_batch, 10, 20)

    compress_batch(get_files(buffer_dir), data, len(result))

    rmtree(buffer_dir)

    rename(output_dir)
