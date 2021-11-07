import tensorflow as tf


def bag_of_words(word_ids, name, vocabulary_size, embedding_size, device):
    with tf.compat.v1.variable_scope(name):
        with tf.device(device):
            word_embeddings = tf.compat.v1.get_variable("word_embeddings", [vocabulary_size, embedding_size])

        return tf.nn.embedding_lookup_sparse(params=word_embeddings, sp_ids=word_ids, sp_weights=None, combiner='mean')
