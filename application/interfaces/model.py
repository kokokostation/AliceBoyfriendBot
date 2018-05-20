import tensorflow as tf


class Model:
    def __init__(self, model_container, model, flavor):
        self.flavor = flavor
        self.model_container = model_container
        self.sparsifiers = self.model_container.read('sparsifiers')

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model_box = model.make_model()

            self.sess = self.model_container.read_model()


class TowersModel(Model):
    def context_embedding(self, context):
        raise NotImplementedError()

    def reply_embedding(self, batch):
        raise NotImplementedError()
