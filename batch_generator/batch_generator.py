from functools import partial

from batch_generator.utils import cycle, transpose


class BatchGenerator:
    def __init__(self, iterator, model_container, flavor, infinite=True, batch_size=None):
        self.batch_size = model_container.model_params(True)['batch_size'] \
            if batch_size is None else batch_size
        self.sparsifiers = model_container.read('sparsifiers')
        self.iterator = iterator
        self.fn = cycle if infinite else iter
        self.flavor = flavor
        self.gen = None

    def __iter__(self):
        self.gen = self.flavor.batch_generator(self.fn(self.iterator), self.batch_size)

        return self

    def __next__(self):
        if self.gen is None:
            self.gen = self.flavor.batch_generator(self.fn(self.iterator), self.batch_size)

        info, batch = next(self.gen)

        result = self.flavor.apply_tensor(batch, self.sparsifiers)

        return info, result


def func_batch_generator(*args, **kwargs):
    return partial(BatchGenerator, *args, **kwargs)
