import numpy as np

from application.interfaces.vector_storage import AggregatingVectorStorage, RestrictingStorage, \
    ScoresVectorStorage


class BaseNumpyStorage(ScoresVectorStorage):
    def __init__(self):
        self.vectors = None

    def fit(self, vectors):
        self.vectors = vectors['vectors']
        return self

    def predict_scores(self, vector):
        return np.dot(self.vectors, vector.T)


class NumpyStorage(BaseNumpyStorage, AggregatingVectorStorage):
    def __init__(self, agg_func=np.argmax):
        BaseNumpyStorage.__init__(self)
        AggregatingVectorStorage.__init__(self, agg_func)


class RestrictingNumpyStorage(BaseNumpyStorage, RestrictingStorage):
    def __init__(self, history_len):
        BaseNumpyStorage.__init__(self)
        RestrictingStorage.__init__(self, history_len)
