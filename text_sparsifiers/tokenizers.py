from nltk.tokenize import TweetTokenizer
from itertools import chain
from functools import partial


words = TweetTokenizer().tokenize


def character_ngram(text, n=3):
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def filling(text, n=3, fill='#'):
    return [text + fill * (n - len(text))]


def sequential_tokenizer(tokenizers):
    def new_tokenizer(text):
        tokens = tokenizers[0](text)

        for tokenizer in tokenizers[1:]:
            tokens = chain.from_iterable(map(tokenizer, tokens))

        return list(tokens)

    return new_tokenizer


def make_filling_character_ngram(n=3, fill='#'):
    return sequential_tokenizer([partial(filling, n=n, fill=fill),
                                 partial(character_ngram, n=n)])


filling_words = sequential_tokenizer([lambda text: ['#' if not text else text], words])