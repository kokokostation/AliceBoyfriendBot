import tensorflow as tf

from embedders.convolutional import convolutional_embedder


def block(input_tensor, conv_data):
    tensors = []

    for i, (out_dim, size) in enumerate(conv_data):
        with tf.compat.v1.variable_scope('conv_{}'.format(i)):
            w = tf.compat.v1.get_variable('W', (size, input_tensor.shape[-1], out_dim), tf.float32,
                                          tf.compat.v1.keras.initializers.he_uniform())

            output = tf.nn.conv1d(input=input_tensor, filters=w, stride=1, padding='SAME')
            output = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12)(output)
            output = tf.nn.relu(output)

        tensors.append(output)

    output = tf.concat(tensors, axis=2)

    return output


@convolutional_embedder
def plain_convolutional_embedder(embeddings, mp):
    output = embeddings

    for i, layer in enumerate(mp['convolutions']):
        with tf.compat.v1.variable_scope('layer_{}'.format(i)):
            output = block(output, layer)

    return output
