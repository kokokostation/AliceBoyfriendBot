import tensorflow as tf


def tower(layer_input, hiddens, dropout=None, training=True):
    output = layer_input
    if dropout is None:
        dropout = [0] * len(hiddens)

    if len(hiddens) != 0:
        for h, rate in zip(hiddens[:-1], dropout[:-1]):
            output = tf.layers.dropout(output, rate, training=training)
            output = tf.layers.dense(output, h, activation=tf.nn.relu)

        output = tf.layers.dropout(output, dropout[-1], training=training)
        output = tf.layers.dense(output, hiddens[-1])

    return output


def normalized_tower(layer_input, hiddens, dropout, training):
    return tf.nn.l2_normalize(tower(layer_input, hiddens, dropout, training), axis=1)
