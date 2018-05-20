import tensorflow as tf
from functools import wraps

from model.utils import apply_mask
from embedders.flavor_embedder import flavor_embedder


def convolutional_embedder(encode):
    @wraps(encode)
    @flavor_embedder
    def wrapped(embeddings, sent_lens, mp):
        embeddings = apply_mask(embeddings, sent_lens)
        rep = encode(embeddings, mp)
        rep = apply_mask(rep, sent_lens)
        rep = tf.reduce_max(rep, axis=1)

        return rep

    return wrapped
