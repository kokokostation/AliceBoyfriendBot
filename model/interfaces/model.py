import tensorflow as tf
import tensorflow_addons as tfa

from batch_generator.flavors import Recurrent
from encoders.tower import tower
from model.utils import dense_towers
from train_utils.utils import make_feed_dict


class BaseModel:
    def __init__(self, model_container, train=True):
        self.model_params = model_container.model_params(train)
        self.placeholders = None

    def make_placeholders_helper(self):
        raise NotImplementedError()

    def make_placeholders(self):
        if self.placeholders is None:
            self.placeholders = self.make_placeholders_helper()

        return self.placeholders

    def make_model_helper(self, batch):
        raise NotImplementedError()

    def make_model_box(self, placeholders, *args):
        raise NotImplementedError()

    def make_model(self):
        batch = self.make_placeholders()

        return self.make_model_box(batch, self.make_model_helper(batch))


class ModelBox:
    def __init__(self, placeholders):
        self.placeholders = placeholders

    def make_loss(self, loss):
        raise NotImplementedError()

    def run(self, sess, ops, batch, index=None):
        index = slice(None) if index is None else index

        return sess.run(ops, feed_dict=make_feed_dict(self.placeholders[index], batch))


class SiameseModelBox(ModelBox):
    def __init__(self, placeholders, towers):
        ModelBox.__init__(self, placeholders)

        self.context_tower, self.reply_tower = towers

    def make_loss(self, loss):
        return loss(self.context_tower, self.reply_tower)


class RankingModelBox(ModelBox):
    def __init__(self, placeholders, output):
        ModelBox.__init__(self, placeholders)

        self.output = output

    def make_loss(self, loss):
        return loss(self.output)


class SiameseModelBoxIntruder(BaseModel):
    def make_model_box(self, placeholders, *args):
        return SiameseModelBox(placeholders, *args)


class RankingModelBoxIntruder(BaseModel):
    def make_model_box(self, placeholders, *args):
        return RankingModelBox(placeholders, *args)


class EmbedderModel(BaseModel):
    def make_embedder(self, item, name, mp, typ):
        raise NotImplementedError()

    def make_embedder_helper(self, item, index, typ):
        return self.make_embedder(item, '{}_{}'.format(typ, index), self.model_params[typ], typ)

    def make_towers_helper(self, context_emb, reply_emb):
        return context_emb, reply_emb

    def make_model_helper(self, batch):
        context_embeddings = [self.make_embedder_helper(item, i, 'context')
                              for i, item in enumerate(batch[:-1])]

        context_emb = tf.concat(context_embeddings, axis=1)
        reply_emb = self.make_embedder_helper(batch[-1], '', 'reply')

        return self.make_towers_helper(context_emb, reply_emb)


class TowersModel(EmbedderModel, SiameseModelBoxIntruder):
    def make_towers_helper(self, context_emb, reply_emb):
        return dense_towers(self.model_params, context_emb, reply_emb)


def make_placeholders(flavor):
    return Recurrent.make_flavor(
        tf.compat.v1.placeholder(tf.int32, [None, None]),
        tf.compat.v1.sparse_placeholder(tf.int32, [None, None]),
        flavor) + \
           [tf.compat.v1.placeholder(tf.int32, [2]), tf.compat.v1.placeholder(tf.int32, [None])]


class FusedModel(BaseModel):
    def make_placeholders_helper(self):
        context_len = self.model_params['context_len']

        return [make_placeholders(self.model_params[key]['flavor'])
                for key, times in [('context', context_len), ('reply', 1)]
                for _ in range(times)]


class UniModel(BaseModel):
    def make_placeholders_helper(self):
        return [tf.compat.v1.sparse_placeholder(tf.int32, [None, None])
                for _ in range(self.model_params['context_len'] + 1)]


class FlavorTowersModel(TowersModel, FusedModel):
    pass


class RankingModel(RankingModelBoxIntruder):
    def make_context(self, context, context_params):
        raise NotImplementedError()

    def make_reply(self, context_outputs, model_params, reply):
        raise NotImplementedError()

    def make_model_helper(self, batch):
        multiplier = self.model_params['multiplier']
        context_params = self.model_params['context']
        reply_params = self.model_params['reply']
        context, reply = batch[:-1], batch[-1]

        context_outputs = self.make_context(context, context_params)
        context_outputs = [tfa.seq2seq.tile_batch(output, multiplier=multiplier)
                           for output in context_outputs]

        output = self.make_reply(context_outputs, self.model_params, reply)

        output = tower(output, reply_params['hiddens'] + [1], reply_params.get('dropout'),
                       self.model_params['train'])

        output = tf.reshape(output, (-1, multiplier))

        return output
