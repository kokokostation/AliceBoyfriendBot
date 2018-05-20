import numpy as np

from application.interfaces.model import Model
from batch_generator.batch_generator import BatchGenerator


class RankingApplicationModel(Model):
    def __init__(self, model_container, model, flavor):
        Model.__init__(self, model_container, model, flavor)

    def rank(self, queries):
        bg = BatchGenerator(queries, self.model_container, self.flavor, False, len(queries))

        _, batch = next(iter(bg))

        answers = self.model_box.run(self.sess, self.model_box.output, batch)

        return [[query[1][i] for i in index] for query, index in zip(queries, np.argsort(-answers))]
