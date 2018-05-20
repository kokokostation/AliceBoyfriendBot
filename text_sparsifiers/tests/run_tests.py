import pytest

from text_sparsifiers.tokenizers import words, character_ngram, word_ngram
from text_sparsifiers.sparsifiers import Sparsifier


def test_words():
    assert words('Hello, world!:)') == ['Hello', ',', 'world', '!', ':)']


def test_character_ngram():
    assert character_ngram('Hello') == ['Hel', 'ell', 'llo']


def test_word_ngram():
    assert word_ngram('Hello, world!') == [['Hello', ','], [',', 'world'], ['world', '!']]


def test_sparsifier_with_limit():
    sparsifier = Sparsifier(words, 2)

    result = sparsifier.fit_transform(['Hello Hello, world!!!'])

    assert result == [[1, 1, 2, 2, 0, 0, 0]]


def test_sparsifier_without_limit():
    sparsifier = Sparsifier(words)

    result = sparsifier.fit_transform(['Hello Hello, world!!!'])

    inv_vocab = {value: key for key, value in sparsifier._vocab.items()}

    assert ' '.join([inv_vocab[a] for a in result[0]]) == 'Hello Hello , world ! ! !'
