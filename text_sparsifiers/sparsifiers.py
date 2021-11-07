from collections import Counter, defaultdict
from itertools import chain

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from batch_generator.mappers import message_mapper
from text_sparsifiers.utils import iter_map


class Sparsifier(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, word_limit=-1, word_limit_mode='absolute'):
        self.tokenizer = tokenizer
        self.word_limit = word_limit
        self.word_limit_mode = word_limit_mode

        self.reset()

    def reset(self):
        self.vocab = {}
        self.inv_vocab = None

    def tokenize(self, texts, fit_tokenizer=None):
        tokenizer = self.tokenizer if fit_tokenizer is None else fit_tokenizer

        return map(tokenizer, texts)

    @property
    def vocabulary_size(self):
        return len(self.vocab) + 2

    @property
    def null(self):
        return len(self.vocab) + 1

    @property
    def unk(self):
        return len(self.vocab)

    def fit(self, X, fit_transform=False, fit_tokenizer=None):
        self.reset()

        tokenized = self.tokenize(X, fit_tokenizer)
        tokenized_chain = chain.from_iterable(tokenized)

        if self.word_limit == -1:
            tokens = set(tokenized_chain)
        else:
            cntr = Counter(tokenized_chain)

            if self.word_limit_mode == 'absolute':
                tokens = [a for a, _ in cntr.most_common(self.word_limit)]
            elif self.word_limit_mode == 'occurrences':
                tokens = [a for a, b in cntr.items() if b >= self.word_limit]
            else:
                raise Exception('Incorrect word_limit_mode')

        self.vocab = dict(zip(tokens, range(len(tokens))))

        if fit_transform:
            return tokenized
        else:
            return self

    def transform(self, X=None, tokenized=None):
        if tokenized is None:
            tokenized = self.tokenize(X)

        return [[self.vocab.get(token, self.unk) for token in text] for text in tokenized]

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, self.fit(X, fit_transform=True))

    def prepare_inv_transform(self):
        if not hasattr(self, 'inv_vocab'):
            self.inv_vocab = None

        if self.inv_vocab is None:
            self.inv_vocab = np.ndarray((self.vocabulary_size,), dtype=np.object)

            self.inv_vocab[self.unk] = '<unk>'
            self.inv_vocab[self.null] = '<null>'

            for token, index in self.vocab.items():
                self.inv_vocab[index] = token

    def inv_transform(self, X):
        self.prepare_inv_transform()

        return [[self.inv_vocab[index] for index in sent] for sent in X]


def make_sparsifiers(iterator, tokenizer, fit_tokenizer=None, vocabulary_size=None, modes=None):
    iterator = iter_map(message_mapper, iterator)

    if vocabulary_size is None:
        vocabulary_size = defaultdict(lambda: -1)

    if modes is None:
        modes = defaultdict(lambda: 'absolute')

    sparsifiers = {key: Sparsifier(tokenizer, vocabulary_size[key], modes[key])
                   for key in ['context', 'reply']}
    sparsifiers['reply'].fit((msgs[-1] for msgs in iterator), fit_tokenizer)
    sparsifiers['context'].fit((msg for msgs in iterator for msg in msgs[:-1]), fit_tokenizer)

    return sparsifiers


def make_sparsifiers_anyway(model_container, train_iterator, words):
    try:
        sparsifiers = model_container.read('sparsifiers')
    except:
        sparsifiers = make_sparsifiers(train_iterator, words)
        model_container.write('sparsifiers', sparsifiers)

    return sparsifiers
