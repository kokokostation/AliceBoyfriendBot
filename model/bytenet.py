from embedders.bytenet import bytenet_embedder
from model.interfaces.model import FlavorTowersModel


class BytenetModel(FlavorTowersModel):
    def make_embedder(self, item, name, mp, typ):
        return bytenet_embedder(item, name, mp)
