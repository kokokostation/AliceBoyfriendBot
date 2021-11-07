import tensorflow as tf
import tensorflow_addons as tfa


def precision(logits):
    return tf.reduce_mean(input_tensor=tfa.seq2seq.hardmax(logits)[:, 0])


def stochastic_precision(logits):
    return tf.reduce_mean(input_tensor=tf.cast(tf.greater(logits[:, 0], logits[:, 1]), tf.float32))
