import tensorflow as tf

from embedders.convolutional import convolutional_embedder


def block(input_tensor, conv_data):
    tensors = []

    for i, (out_dim, size) in enumerate(conv_data):
        with tf.variable_scope('conv_{}'.format(i)):
            w = tf.get_variable('W', (size, input_tensor.shape[-1], out_dim), tf.float32,
                                tf.keras.initializers.he_uniform())

            output = tf.nn.conv1d(input_tensor, w, 1, 'SAME')
            output = tf.contrib.layers.layer_norm(output, activation_fn=tf.nn.relu)

        tensors.append(output)

    output = tf.concat(tensors, axis=2)

    return output


@convolutional_embedder
def plain_convolutional_embedder(embeddings, mp):
    output = embeddings

    for i, layer in enumerate(mp['convolutions']):
        with tf.variable_scope('layer_{}'.format(i)):
            output = block(output, layer)

    return output
