import tensorflow as tf

from application.interfaces.model import TowersModel


class ApplicationModel(TowersModel):
    def reply_embedding(self, batch):
        return self.model_box.run(self.sess, self.model_box.reply_tower, batch, -1)

    def context_embedding_batch(self, batch):
        return self.model_box.run(self.sess, self.model_box.context_tower, batch)

    def context_embedding(self, context):
        context = [self.flavor.tensor([item], self.sparsifiers['context'], 'context')
                   for item in context]

        return self.model_box.run(self.sess, self.model_box.context_tower,
                                  context, slice(None, -1))[0]

    def apply(self, batch):
        return self.model_box.run([self.model_box.context_tower,
                                   self.model_box.reply_tower],
                                  batch)
