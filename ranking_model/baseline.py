import tensorflow as tf

from model.baseline import BaseBaseline
from model.interfaces.model import RankingModel, UniModel


class RankingBaseline(RankingModel, UniModel):
    def __init__(self, model_container, train=True):
        RankingModel.__init__(self, model_container, train)

        self.base_baseline = BaseBaseline(model_container, train)
        self.reply_emb = None

    def make_context(self, context, context_params):
        context_emb, self.reply_emb = self.base_baseline.make_model_helper(
            self.make_placeholders())

        return [context_emb]

    def make_reply(self, context_outputs, model_params, reply):
        return tf.concat([context_outputs[0], self.reply_emb], axis=1)
