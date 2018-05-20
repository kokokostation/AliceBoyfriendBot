import tensorflow as tf


def precision(logits):
    return tf.reduce_mean(tf.contrib.seq2seq.hardmax(logits)[:, 0])


def stochastic_precision(logits):
    return tf.reduce_mean(tf.cast(tf.greater(logits[:, 0], logits[:, 1]), tf.float32))