import tensorflow as tf

from embedders.bag_of_words import bag_of_words
from model.interfaces.model import EmbedderModel, UniModel, TowersModel
from embedders.utils import get_device


class BaseBaseline(EmbedderModel, UniModel):
    def make_embedder(self, item, name, mp, typ):
        return bag_of_words(item, name, mp['vocabulary_size'],
                            mp['embedding_size'], get_device(mp))


class Baseline(BaseBaseline, TowersModel):
    pass
