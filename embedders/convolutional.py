from functools import wraps

import tensorflow as tf

from embedders.flavor_embedder import flavor_embedder
from model.utils import apply_mask


def convolutional_embedder(encode):
    @wraps(encode)
    @flavor_embedder
    def wrapped(embeddings, sent_lens, mp):
        embeddings = apply_mask(embeddings, sent_lens)
        rep = encode(embeddings, mp)
        rep = apply_mask(rep, sent_lens)
        rep = tf.reduce_max(input_tensor=rep, axis=1)

        return rep

    return wrapped
