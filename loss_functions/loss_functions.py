import tensorflow as tf

from loss_functions.utils import pairwise_similarities


def sample_pos_neg_similarities(similarities, batch_size, sampling, skew=1, lam=None):
    neg_ones = (tf.ones_like(similarities) - tf.eye(batch_size))
    if sampling == "uniform":
        neg_examples_probs = neg_ones
    elif sampling == "weighted":
        lam = lam or 100.
        q = tf.subtract(1.0, similarities)  # distances / 2
        sample_weights = tf.minimum(lam, tf.math.reciprocal(q))
        neg_examples_probs = tf.multiply(sample_weights, neg_ones)
    else:
        raise NotImplementedError("sampling {} not implemented".format(sampling))

    neg_examples_ids = tf.reshape(
        tf.random.categorical(logits=tf.math.log(neg_examples_probs) * skew, num_samples=1),
        [-1],
    )
    neg_examples = tf.reduce_sum(
        input_tensor=tf.multiply(
            similarities,
            tf.one_hot(
                neg_examples_ids,
                batch_size
            )
        ),
        axis=1
    )
    pos_examples = tf.reduce_sum(
        input_tensor=tf.multiply(
            similarities,
            tf.eye(batch_size)
        ),
        axis=1
    )
    return pos_examples, neg_examples


def easy_negative_pce(batch_questions, batch_answers, mode='sigmoid', sampling="weighted",
                      lmd=1.0, batch_size=50, lam=None):
    similarities = pairwise_similarities(batch_questions, batch_answers)

    pos_similarities, neg_similarities = sample_pos_neg_similarities(similarities, batch_size, sampling, 1, lam)
    diff = tf.subtract(pos_similarities, neg_similarities)

    if mode == 'sigmoid':
        with tf.compat.v1.variable_scope(
                "easy_negative_pce_sigmoid",
                reuse=tf.compat.v1.AUTO_REUSE,
                initializer=tf.random.normal(shape=()),
        ):
            alpha = tf.compat.v1.get_variable('alpha', dtype=tf.float32)
            beta = tf.compat.v1.get_variable('beta', dtype=tf.float32)
        return -tf.reduce_sum(
            input_tensor=tf.math.log_sigmoid(
                tf.add(
                    beta,
                    tf.scalar_mul(alpha, diff)
                )
            )
        )
    if mode == 'margin':
        with tf.compat.v1.variable_scope(
                "easy_negative_pce_adamargin",
                reuse=tf.compat.v1.AUTO_REUSE,
                initializer=tf.random.normal(shape=()),
        ):
            # alpha = tf.get_variable('alpha', dtype=tf.float32)
            alpha = lmd
            beta = tf.compat.v1.get_variable('beta', dtype=tf.float32)

        return tf.add_n([
            tf.reduce_sum(
                input_tensor=tf.nn.relu(
                    tf.add(
                        alpha,
                        sign * tf.subtract(
                            examples,
                            beta)
                    )
                )
            )
            for examples, sign in [
                (pos_similarities, -1),
                (neg_similarities, 1),
            ]
        ])

    elif mode == 'triplet':
        return tf.reduce_sum(input_tensor=tf.nn.relu(lmd - diff))


def full_pairwise_softmax(batch_questions, batch_answers):
    with tf.compat.v1.variable_scope('fps'):
        similarities = pairwise_similarities(batch_questions, batch_answers)

        alpha = tf.Variable(0.1, dtype=tf.float32, name='alpha')
        similarities = tf.scalar_mul(alpha, similarities)

        return tf.add_n([
            tf.reduce_sum(
                input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.range(tf.shape(input=batch_questions)[0]),
                    logits=logits
                )
            )
            for logits in [similarities, tf.transpose(a=similarities)]
        ])
