import tensorflow as tf

from embedders.convolutional import convolutional_embedder


def res_block(tensor, block, is_first, rate, size, emb_size):
    in_dim = emb_size
    half_dim = in_dim // 2

    with tf.compat.v1.variable_scope('block_{}_{}'.format(block, rate)):
        out = tensor

        if not is_first:
            out = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12)(out)
            out = tf.nn.relu(out)
        out = tf.nn.relu(out)

        w_1 = tf.compat.v1.get_variable('W_1', (1, in_dim, half_dim), tf.float32,
                                        tf.compat.v1.keras.initializers.he_uniform())
        out = tf.nn.conv1d(input=out, filters=w_1, stride=1, padding='SAME')

        out = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12)(out)
        out = tf.nn.relu(out)

        w_2 = tf.compat.v1.get_variable('W_2', (size, half_dim, half_dim), tf.float32,
                                        tf.compat.v1.keras.initializers.he_uniform())
        out = tf.nn.convolution(input=out, filters=w_2, padding='SAME', dilations=[rate])

        out = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-12)(out)
        out = tf.nn.relu(out)

        w_3 = tf.compat.v1.get_variable('W_3', (1, half_dim, in_dim), tf.float32,
                                        tf.compat.v1.keras.initializers.he_uniform())
        b_3 = tf.Variable([0.] * in_dim, name='b_3', dtype=tf.float32)
        out = tf.nn.conv1d(input=out, filters=w_3, stride=1, padding='SAME') + b_3

        out = out + tensor

    return out


@convolutional_embedder
def bytenet_embedder(rep, mp):
    rates, nums, size = mp['rates'], mp['nums'], mp['size']

    for i in range(nums):
        for j, rate in enumerate(rates):
            rep = res_block(rep, i, j == 0, rate, size, rep.shape[-1])

    return rep
