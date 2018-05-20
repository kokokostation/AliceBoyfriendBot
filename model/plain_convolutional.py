from embedders.plain_convolutional import plain_convolutional_embedder
from model.interfaces.model import FlavorTowersModel


class PlainConvolutionalModel(FlavorTowersModel):
    def make_embedder(self, item, name, mp, typ):
        return plain_convolutional_embedder(item, name, mp)