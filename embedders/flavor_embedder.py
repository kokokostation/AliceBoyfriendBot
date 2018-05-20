import tensorflow as tf
from functools import wraps

from embedders.utils import get_device


def flavor_embedder(embedder):
    @wraps(embedder)
    def wrapped(data, name, mp, *args, **kwargs):
        with tf.variable_scope(name):
            flavor = mp['flavor']
            shape, sent_lens = data[-2:]

            with tf.device(get_device(mp)):
                if flavor == 'words':
                    embeddings = word_embedder(data[0], mp)
                elif flavor == 'ngrams':
                    embeddings = ngram_embedder(data[0], shape, mp)
                elif flavor == 'combined':
                    word_ids, ngram_ids = data[:2]
                    w_emb = word_embedder(word_ids, mp)
                    ng_emb = ngram_embedder(ngram_ids, shape, mp)

                    embeddings = tf.concat([w_emb, ng_emb], axis=2)

            return embedder(embeddings, sent_lens, mp, *args, **kwargs)

    return wrapped


def word_embedder(word_ids, mp):
    embedding_matrix = tf.get_variable("word_embeddings", [mp['word_vocabulary_size'],
                                                           mp['word_embedding_size']])

    word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)

    word_embeddings = add_dropout(word_embeddings, mp, 'word')

    return word_embeddings


def ngram_embedder(ngram_ids, shape, mp):
    embedding_matrix = tf.get_variable("ngram_embeddings", [mp['ngram_vocabulary_size'],
                                                            mp['ngram_embedding_size']])

    embeddings = tf.nn.embedding_lookup_sparse(embedding_matrix, ngram_ids, None, combiner='mean')

    shape = tf.concat([shape, (mp['ngram_embedding_size'],)], axis=0)
    embeddings = tf.reshape(embeddings, shape)

    embeddings = add_dropout(embeddings, mp, 'ngram')

    return embeddings


def add_dropout(embeddings, mp, prefix):
    keep_prob = mp.get('{}_keep_prob'.format(prefix))

    if keep_prob is not None and mp['train']:
        noise_shape = tf.concat([tf.shape(embeddings)[:-1], (1,)], axis=0)

        embeddings = tf.nn.dropout(embeddings, keep_prob, noise_shape)

    return embeddings