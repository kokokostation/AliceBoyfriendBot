import tensorflow as tf

from encoders.tower import normalized_tower


def dense_towers(model_params, context_emb, reply_emb):
    towers = {
        'context': context_emb,
        'reply': reply_emb
    }

    for typ, tower in towers.items():
        mp = model_params[typ]

        towers[typ] = normalized_tower(tower, mp['hiddens'], mp.get('dropout'), mp['train'])

    return towers['context'], towers['reply']


def apply_mask(rep, sent_lens):
    maxlen = tf.shape(input=rep)[1]
    emb_size = rep.shape[-1]

    mask = tf.cast(tf.sequence_mask(sent_lens, maxlen), tf.float32)
    mask = tf.stack([mask] * emb_size, axis=2)

    return rep * mask
