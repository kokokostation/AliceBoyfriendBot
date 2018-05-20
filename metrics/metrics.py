import tensorflow as tf

from loss_functions.utils import pairwise_similarities


def precision_at_one(batch_questions, batch_answers):
    similarities = pairwise_similarities(batch_questions, batch_answers)
    batch_size = tf.shape(batch_questions)[0]
    neg_examples_probs = (tf.ones_like(similarities) - tf.eye(batch_size)) / (tf.cast(batch_size, tf.float32) - 1)
    neg_examples_ids = tf.reshape(tf.multinomial(tf.log(neg_examples_probs),
                                                 1),
                                  [-1])
    neg_examples = tf.reduce_sum(tf.multiply(similarities,
                                             tf.one_hot(neg_examples_ids,
                                                        batch_size)),
                                 axis=1)
    pos_examples = tf.reduce_sum(tf.multiply(similarities,
                                             tf.eye(batch_size)),
                                 axis=1)
    return tf.reduce_mean(tf.cast(tf.greater(pos_examples, neg_examples), tf.float32))


def det_precision_at_one(batch_questions, batch_answers):
    similarities = pairwise_similarities(batch_questions, batch_answers)
    batch_size = tf.shape(batch_questions)[0]

    rng = tf.range(batch_size)
    inds = tf.transpose([rng, (rng + 1) % batch_size])
    negatives = tf.gather_nd(similarities, inds)

    positives = tf.diag_part(similarities)

    result = tf.reduce_mean(tf.cast(tf.greater(positives, negatives), tf.float32))

    return result
