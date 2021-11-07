import tensorflow as tf


def cross_entropy(ranking_matrix):
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.fill(tf.shape(input=ranking_matrix)[:1], 0),
            logits=ranking_matrix))
