import numpy as np
import tensorflow as tf
import os
from itertools import chain


def plain_batch_generator(gen, batch_size, callback=None):
    if callback is None:
        callback = lambda x: x

    res = []

    for item in gen:
        res.append(item)

        if len(res) == batch_size:
            yield callback(res)

            res = []


def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int32)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int32)

    return tf.SparseTensorValue(indices=indices, values=values, dense_shape=shape)


def cycle(iterator):
    while True:
        for item in iterator:
            yield item


def get_batch(gen, batch_size):
    return [item for item, _ in zip(gen, range(batch_size))]


def get_paths(folder):
    return [os.path.join(folder, fname) for fname in os.listdir(folder)]


def get_random_paths(folder, file_limit=None):
    paths = get_paths(folder)

    if file_limit is not None:
        np.random.shuffle(paths)
        paths = paths[:file_limit]

    return paths


def align(lists, fill):
    max_len = max(map(len, lists))
    result = np.zeros((len(lists), max_len), dtype=np.int) + fill

    for i, l in enumerate(lists):
        result[i, :len(l)] = l

    return result


def align_list(l, lens, fill):
    max_len = max(lens)
    inds = np.cumsum(lens)
    result = []

    for i, ii in enumerate(inds):
        begin = 0 if i == 0 else inds[i - 1]
        result.extend(l[begin:inds[i]] + [fill] * (max_len - (inds[i] - begin)))

    return result


def join_lists(lists, sep):
    result = []

    for l in lists:
        result.extend(l)
        result.append(sep)

    return result


def transpose(lists):
    return [[lists[i][j] for i, _ in enumerate(lists)] for j, _ in enumerate(lists[0])]


def flatten_to_str(l):
    if type(l) == str:
        return [l]
    else:
        return chain.from_iterable(map(flatten_to_str, l))
