import numpy as np

from application.interfaces.vector_storage import AggregatingVectorStorage, \
    ScoresVectorStorage, RestrictingStorage


class BasePriorStorage(ScoresVectorStorage):
    def __init__(self, prior_factor):
        self.info = None
        self.prior_factor = prior_factor

    def fit(self, info):
        self.info = info
        priors = np.log(self.info['priors'] + 1)
        priors -= np.min(priors)
        max_prior = np.max(priors)
        if max_prior != 0:
            priors /= max_prior
        self.info['priors'] = priors

        return self

    def predict_scores(self, vector):
        vectors, priors = self.info['vectors'], self.info['priors']
        scores = np.dot(vectors, vector.T) + self.prior_factor * priors

        return scores


class PriorStorage(BasePriorStorage, AggregatingVectorStorage):
    def __init__(self, prior_factor, agg_func=np.argmax):
        BasePriorStorage.__init__(self, prior_factor)
        AggregatingVectorStorage.__init__(self, agg_func)


class RestrictingPriorStorage(BasePriorStorage, RestrictingStorage):
    def __init__(self, prior_factor, history_len):
        BasePriorStorage.__init__(self, prior_factor)
        RestrictingStorage.__init__(self, history_len)
