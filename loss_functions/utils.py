import tensorflow as tf


def pairwise_similarities(batch_questions, batch_answers):
    similarities = tf.matmul(
        batch_questions,
        batch_answers,
        adjoint_b=True,  # transpose second matrix
    )
    return similarities
