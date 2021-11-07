import tensorflow as tf

from embedders.flavor_embedder import flavor_embedder


@flavor_embedder
def lstm_embedder(embeddings, sent_lens, mp):
    lstm_flavor = mp['lstm_flavor']
    if lstm_flavor == 'bidirectional':
        cells = [tf.compat.v1.nn.rnn_cell.BasicLSTMCell(mp['hid_size']) for _ in range(2)]
        outputs, (last_fw, last_bw) = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            cells[0], cells[1], embeddings, sent_lens, dtype=tf.float32)
        output = tf.concat([last_fw.c, last_bw.c], axis=1)
    elif lstm_flavor == 'plain':
        cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(mp['hid_size'])
        outputs, last = tf.compat.v1.nn.dynamic_rnn(cell, embeddings, sent_lens, dtype=tf.float32)
        output = last.c

    return output
