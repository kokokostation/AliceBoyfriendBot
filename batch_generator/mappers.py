from functools import partial, wraps
import numpy as np
import tensorflow as tf

from batch_generator.prior import PriorGetter
from batch_generator.dir import DirIterator
from batch_generator.utils import cycle


def agnostic_mapper(mapper):
    @wraps(mapper)
    def wrapped(model_container, **kwargs):
        return partial(mapper, **kwargs)

    return wrapped


def message_mapper(item):
    return [a['processed_body'] for a in item]

def uid_mapper(item):
    return item[-1]['author']

def common_reply_mapper(get_prior, reply_extractor, item, orig_key='body'):
    orig_message, message = [reply_extractor(item)[key] for key in [orig_key, 'processed_body']]

    return (orig_message, get_prior(message)), [message]


plain_reply_mapper = agnostic_mapper(partial(common_reply_mapper, lambda _: 1, lambda item: item[-1]))


def prior_reply_mapper(mc, **kwargs):
    return partial(common_reply_mapper, PriorGetter(mc), lambda item: item, **kwargs)


def fusion_message_mapper(item):
    msgs = message_mapper(item)

    return [msgs[:-1], [msgs[-1]]]


def reply_fusion(mapper):
    @wraps(mapper)
    def wrapped(mc, **kwargs):
        ready_mapper = mapper(mc, **kwargs)

        def wwrapped(*args, **kwargs):
            om, m = ready_mapper(*args, **kwargs)

            return om, [m]

        return wwrapped

    return wrapped


plain_fusion_reply_mapper = reply_fusion(plain_reply_mapper)
prior_fusion_reply_mapper = reply_fusion(prior_reply_mapper)


@agnostic_mapper
def train_mapper(item, mm=message_mapper):
    return None, mm(item)


@agnostic_mapper
def rank_preparation_mapper(item, mm=message_mapper):
    return message_mapper(item), mm(item)

 
@agnostic_mapper
def uid_train_mapper(item, mm=message_mapper, ranking=False):
    res = mm(item)
    uid = uid_mapper(item)
   
    return res if ranking else None, [uid] + res


@agnostic_mapper
def test_mapper(item, mm=message_mapper):
    orig_message = ['###'.join(msg['processed_body'] for msg in item[:-1]),
                    item[-1]['processed_body']]

    return orig_message, mm(item)


@agnostic_mapper
def rank_fused_train_mapper(item):
    context, replies = item

    return None, [context, [[x] for x in replies]]


@agnostic_mapper
def rank_train_mapper(item):
    context, replies = item

    return None, context + [replies]


class Percentage:
    def __init__(self, value):
        self.value = value
        self.tf_value = tf.Variable(value, dtype=tf.float32, trainable=False)

    def set_value(self, trainer, value):
        self.value = value
        trainer.sess.run(self.tf_value.assign(value))


def weak_rank_train_mapper(replies_dir, percentage, base_mapper):
    gen = cycle(DirIterator.from_data_folder(replies_dir))
    rm = base_mapper(None)

    def mapper(item):
        context, replies = item

        inds = np.where(np.random.binomial(1, percentage.value, len(replies) - 1))[0] + 1

        for i, reply in zip(inds, gen):
            replies[i] = reply['processed_body']

        return rm((context, replies))

    return mapper
