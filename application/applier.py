from collections import Iterable

from application.model import ApplicationModel
from application.context_preparer import ContextPreparer
from dataset.reddit import convert
from application.numpy_storage import NumpyStorage
from batch_generator.dir import DirIterator
from batch_generator.batch_generator import BatchGenerator
from application.utils import make_replies
from train_utils.container import ModelContainer
import json


class BaseApplier:
    def __init__(self, model, vector_storage, context_preparer, data):
        self.model = model
        self.vector_storage = vector_storage
        self.context_preparer = context_preparer

        if isinstance(data, ModelContainer):
            self.data = data.read('applier')
        elif isinstance(data, Iterable):
            self.data = make_replies(model, data)
        else:
            self.data = data

        self.vector_storage.fit(self.data['vectors'])

    def reply_helper(self, context, replies):
        raise NotImplementedError()

    def reply(self, context):
        plain_context = self.context_preparer(context)

        context_emb = self.model.context_embedding(plain_context)

        index = self.vector_storage.predict(context_emb)

        dr = self.data['replies']
        replies = [dr[i] for i in index] if isinstance(index, list) else dr[index]

        return self.reply_helper(context, replies)

    def to_pickle(self, model_container):
        model_container.write('applier', self.data)


class Applier(BaseApplier):
    def reply_helper(self, context, replies):
        return replies


class RankingApplier(BaseApplier):
    def __init__(self, plain_model, vector_storage, plain_context_preparer,
                 ranking_context_preparer, data, ranking_model, container, verbose=False):
        BaseApplier.__init__(self, plain_model, vector_storage, plain_context_preparer, data)

        self.ranking_model = ranking_model
        self.ranking_context_preparer = ranking_context_preparer
        self.container = container
        self.verbose = verbose

    def reply_helper(self, context, replies):
        ranking_context = self.ranking_context_preparer(context)

        ranked_replies = self.ranking_model.rank([[ranking_context, replies]])[0]

        reply = self.container.get_next_entry(ranked_replies)

        if self.verbose:
            return 'REPLY:\n{}\nMODEL_HYPOS:\n{}\nRANKED_HYPOS\n{}'.format(
                reply, replies, ranked_replies)
        else:
            return reply


def make_applier(model_container, model, flavor, vector_storage=None, context_preparer=None,
                 train_data_dir=None, file_limit=None):
    if context_preparer is None:
        context_preparer = ContextPreparer(convert)
    if vector_storage is None:
        vector_storage = NumpyStorage()

    am = ApplicationModel(model_container, model, flavor)

    if model_container.exists('applier'):
        ap = Applier.from_pickle(am, vector_storage, context_preparer, model_container)
    else:
        iterator = DirIterator.from_data_folder(train_data_dir, file_limit)
        gen = BatchGenerator(iterator, model_container, flavor, infinite=False)

        ap = Applier.from_gen(am, vector_storage, context_preparer, gen)

        ap.to_pickle(model_container)

    return ap


