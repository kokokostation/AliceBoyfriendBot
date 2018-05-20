import tensorflow as tf


def bag_of_words(word_ids, name, vocabulary_size, embedding_size, device):
    with tf.variable_scope(name):
        with tf.device(device):
            word_embeddings = tf.get_variable("word_embeddings", [vocabulary_size, embedding_size])

        return tf.nn.embedding_lookup_sparse(word_embeddings, word_ids, None, combiner='mean')
