from functools import partial
from itertools import chain

import numpy as np

from batch_generator.interfaces.flavor import MapperFlavor
from batch_generator.mappers import train_mapper, plain_reply_mapper, test_mapper, \
    fusion_message_mapper, prior_reply_mapper, plain_fusion_reply_mapper, \
    prior_fusion_reply_mapper, rank_fused_train_mapper, rank_train_mapper, \
    rank_preparation_mapper, weak_rank_train_mapper, uid_train_mapper
from batch_generator.utils import get_batch, sparse_tuple_from, align, align_list, join_lists, \
    transpose, flatten_to_str


class Plain(MapperFlavor):
    def batch_generator_helper(self, gen, batch_size):
        while True:
            batch = get_batch(gen, batch_size)
            if batch is None:
                break

            yield batch


    def tensor(self, data, sparsifier, typ):
        return sparse_tuple_from(sparsifier.transform(data))


class Recurrent(MapperFlavor):
    def __init__(self, block_factor, mapper, train, model_container, **kwargs):
        MapperFlavor.__init__(self, mapper(model_container, **kwargs))

        self.block_factor = block_factor
        self.mp = model_container.model_params(train)

    def batch_generator_helper(self, gen, batch_size):
        batch_optimization = self.mp.get('batch_optimization')
        if batch_optimization is None:
            batch_optimization = self.mp['train']

        while True:
            if batch_optimization:
                block = get_batch(gen, batch_size * self.block_factor)
                block = sorted(block, key=lambda x:
                np.mean([len(el) for el in flatten_to_str(x[1])]))
                slices = [slice(i, i + batch_size) for i in range(0, len(block), batch_size)]
                np.random.shuffle(slices)

                for sl in slices:
                    yield block[sl]
            else:
                batch = get_batch(gen, batch_size)
                if batch is None:
                    break

                yield batch

    @staticmethod
    def make_flavor(words, ngrams, flavor):
        return {
            'words': [words],
            'ngrams': [ngrams],
            'combined': [words, ngrams]
        }[flavor]

    def prepare_data(self, sparsifier, data):
        return list(sparsifier.tokenize(data))

    def prepare_words(self, sparsifier, data):
        return sparsifier.transform(tokenized=data)

    def prepare_ngrams(self, sparsifier, data):
        return sparsifier.transform(chain.from_iterable(data))

    def get_ls(self, data):
        sent_lens = list(map(len, data))
        shape = len(data), max(sent_lens)

        return sent_lens, shape

    def tensor(self, data, sparsifier, typ):
        flavor = self.mp[typ]['flavor']

        data = self.prepare_data(sparsifier['words'], data)
        sent_lens, shape = self.get_ls(data)

        words, ngrams = None, None

        if flavor in ['words', 'combined']:
            words = self.prepare_words(sparsifier['words'], data)
            words = align(words, sparsifier['words'].null)

        if flavor in ['ngrams', 'combined']:
            processed = self.prepare_ngrams(sparsifier['ngrams'], data)
            ngrams = align_list(processed, sent_lens, [sparsifier['ngrams'].null])
            ngrams = sparse_tuple_from(ngrams)

        return Recurrent.make_flavor(words, ngrams, flavor) + [shape, sent_lens]


class RecurrentFused(Recurrent):
    def prepare_data(self, sparsifier, data):
        return [Recurrent.prepare_data(self, sparsifier, item) for item in transpose(data)]

    def get_ls(self, data):
        sent_lens = [sum(len(item[i]) for item in data) + len(data)
                     for i, _ in enumerate(data[0])]
        shape = len(data[0]), max(sent_lens)

        return sent_lens, shape

    def prepare_words(self, sparsifier, data):
        data = [Recurrent.prepare_words(self, sparsifier, item) for item in data]

        return [join_lists([item[i] for item in data], sparsifier.null)
                for i, _ in enumerate(data[0])]

    def prepare_ngrams(self, sparsifier, data):
        tokens = [Recurrent.prepare_ngrams(self, sparsifier, item) for item in data]

        ends = [np.cumsum(list(map(len, item))).tolist() for item in data]
        all_borders = zip(*[zip([0] + item[:-1], item) for item in ends])

        posts = [join_lists([item[begin:end]
                             for item, (begin, end) in zip(tokens, borders)],
                            [sparsifier.null])
                 for borders in all_borders]

        return list(chain.from_iterable(posts))


def make_ranking_flavor(flavor_cls):
    class Ranking(flavor_cls):
        def tensor(self, data, sparsifier, typ):
            if typ == 'reply':
                return super().tensor(list(chain.from_iterable(data)), sparsifier, typ)
            else:
                return super().tensor(data, sparsifier, typ)

    return Ranking


RankingFused = make_ranking_flavor(RecurrentFused)
RankingPlain = make_ranking_flavor(Plain)

PLAIN_TRAIN = Plain(train_mapper(None))
PLAIN_REPLY = Plain(plain_reply_mapper(None))
PLAIN_PRIOR_REPLY = lambda mc: Plain(prior_reply_mapper(mc))
PLAIN_TEST = Plain(test_mapper(None))

RECURRENT_TRAIN = partial(Recurrent, 10, train_mapper, True)
RECURRENT_REPLY = partial(Recurrent, 10, plain_reply_mapper, False)
RECURRENT_PRIOR_REPLY = partial(Recurrent, 10, prior_reply_mapper, False)
RECURRENT_TEST = partial(Recurrent, 10, test_mapper, False)

RECURRENT_FUSED_TRAIN = partial(RecurrentFused,
                                10,
                                partial(train_mapper, mm=fusion_message_mapper),
                                True)
RECURRENT_FUSED_REPLY = partial(RecurrentFused, 10, plain_fusion_reply_mapper, False)
RECURRENT_FUSED_PRIOR_REPLY = partial(RecurrentFused,
                                      10,
                                      prior_fusion_reply_mapper,
                                      False)
RECURRENT_FUSED_TEST = partial(RecurrentFused,
                               10,
                               partial(test_mapper, mm=fusion_message_mapper),
                               False)
RECURRENT_FUSED_RANKING_PREPARATION = partial(RecurrentFused,
                                              10,
                                              partial(rank_preparation_mapper,
                                                      mm=fusion_message_mapper),
                                              False)
RECURRENT_FUSED_REPLY_FOR_RANKING = partial(RecurrentFused, 10,
                                            partial(prior_fusion_reply_mapper,
                                                    orig_key='processed_body'),
                                            False)

CONTEXT_UID_TRAIN = Plain(uid_train_mapper(None))

RANKING_FUSED_TRAIN = partial(RankingFused, 10, rank_fused_train_mapper, True)
RANKING_PLAIN_TRAIN = RankingPlain(rank_train_mapper(None))
