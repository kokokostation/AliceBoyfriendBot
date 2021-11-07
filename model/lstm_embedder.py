from embedders.lstm import lstm_embedder
from model.interfaces.model import FlavorTowersModel


class LSTMModel(FlavorTowersModel):
    def make_embedder(self, item, name, mp, typ):
        return lstm_embedder(item, name, mp)
