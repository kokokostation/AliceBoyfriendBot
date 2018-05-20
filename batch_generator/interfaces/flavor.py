from batch_generator.utils import transpose
from itertools import tee


class Flavor:
    def batch_generator(self, gen, batch_size):
        raise NotImplementedError()

    def tensor(self, data, sparsifier, typ):
        raise NotImplementedError()

    def apply_tensor(self, batch, sparsifiers):
        samples = transpose(batch)

        result = []
        for i, item in enumerate(samples):
            typ = 'reply' if i == len(samples) - 1 else 'context'
            sparsifier = sparsifiers[typ]

            result.append(self.tensor(item, sparsifier, typ))

        return result



class MapperFlavor(Flavor):
    def __init__(self, mapper):
        self.mapper = mapper

    def batch_generator(self, gen, batch_size):
        return map(transpose, self.batch_generator_helper(map(self.mapper, gen), batch_size))

    def batch_generator_helper(self, gen, batch_size):
        raise NotImplementedError()


class FlavorConcater(Flavor):
    def __init__(self, flavors):
        self.flavors = flavors

    def batch_generator(self, gen, batch_size):
        gens = tee(gen, len(self.flavors))
        flavor_gens = [flavor(g, batch_size) for flavor, g in zip(self.flavors, gens)]

        while True:
            data = [next(g) for g in flavor_gens]

            infos, batches = zip(*data)
            batches = list(zip(*batches))

            yield infos, batches

    def tensor(self, data, sparsifier, typ):
        return tuple(zip(transpose(data), self.flavors))
