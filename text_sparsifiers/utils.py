class IteratorWrapper:
    def __init__(self, f, iterator):
        self.f = f
        self.iterator = iterator
        self.gen = None

    def __iter__(self):
        self.gen = iter(self.iterator)

        return self

    def __next__(self):
        return self.f(next(self.gen))


def iter_map(f, iterator):
    return IteratorWrapper(f, iterator)
