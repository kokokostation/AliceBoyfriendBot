import tensorflow as tf

from embedders.convolutional import convolutional_embedder


def res_block(tensor, block, is_first, rate, size, emb_size):
    in_dim = emb_size
    half_dim = in_dim // 2

    with tf.variable_scope('block_{}_{}'.format(block, rate)):
        out = tensor

        if not is_first:
            out = tf.contrib.layers.layer_norm(out)
        out = tf.nn.relu(out)

        w_1 = tf.get_variable('W_1', (1, in_dim, half_dim), tf.float32,
                              tf.keras.initializers.he_uniform())
        out = tf.nn.conv1d(out, w_1, 1, 'SAME')

        out = tf.contrib.layers.layer_norm(out, activation_fn=tf.nn.relu)

        w_2 = tf.get_variable('W_2', (size, half_dim, half_dim), tf.float32,
                              tf.keras.initializers.he_uniform())
        out = tf.nn.convolution(out, w_2, 'SAME', dilation_rate=[rate])

        out = tf.contrib.layers.layer_norm(out, activation_fn=tf.nn.relu)

        w_3 = tf.get_variable('W_3', (1, half_dim, in_dim), tf.float32,
                              tf.keras.initializers.he_uniform())
        b_3 = tf.Variable([0.] * in_dim, name='b_3', dtype=tf.float32)
        out = tf.nn.conv1d(out, w_3, 1, 'SAME') + b_3

        out = out + tensor

    return out


@convolutional_embedder
def bytenet_embedder(rep, mp):
    rates, nums, size = mp['rates'], mp['nums'], mp['size']

    for i in range(nums):
        for j, rate in enumerate(rates):
            rep = res_block(rep, i, j == 0, rate, size, rep.shape[-1])

    return rep
