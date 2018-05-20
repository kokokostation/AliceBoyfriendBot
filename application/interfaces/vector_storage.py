import numpy as np
from collections import deque

from application.utils import k_best


class VectorStorage:
    def fit(self, vectors):
        raise NotImplementedError()

    def predict(self, vector):
        raise NotImplementedError()


class ScoresVectorStorage(VectorStorage):
    def predict_scores(self, vector):
        raise NotImplementedError()

    def predict_helper(self, scores):
        raise NotImplementedError()

    def predict(self, vector):
        return self.predict_helper(self.predict_scores(vector))


class AggregatingVectorStorage(ScoresVectorStorage):
    def __init__(self, agg_func=np.argmax):
        self.agg_func = agg_func

    def predict_helper(self, scores):
        return self.agg_func(scores)


class ReplyContainer:
    def get_next_entry(self, entries):
        return entries[0]


class RestrictingContainer(ReplyContainer):
    def __init__(self, history_len):
        self.previous_answers = deque()
        self.history_len = history_len

    def get_next_entry(self, entries):
        assert len(entries) > self.history_len
        previous_answers_set = set(self.previous_answers)

        for hypo in entries:
            if hypo not in previous_answers_set:
                if len(self.previous_answers) == self.history_len:
                    self.previous_answers.pop()

                self.previous_answers.appendleft(hypo)

                return hypo


class RestrictingStorage(ScoresVectorStorage):
    def __init__(self, history_len):
        VectorStorage.__init__(self)

        self.rc = RestrictingContainer(history_len)

    def predict_helper(self, scores):
        best_inds = k_best(self.rc.history_len + 1, scores)

        return self.rc.get_next_entry(best_inds)




