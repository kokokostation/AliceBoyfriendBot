from multiprocessing_on_dill import Queue, Process


def worker(q, gen):
    for item in gen:
        q.put(item)


class MultiprocessQueue:
    def __init__(self, gens, maxsize):
        self.gens = gens
        self.maxsize = maxsize

        self.reset()

    @staticmethod
    def from_gen_maker(make_gen, njobs, maxsize):
        return MultiprocessQueue([make_gen() for _ in range(njobs)], maxsize)

    def reset(self):
        self.queue = Queue(maxsize=self.maxsize)
        self.ps = []

    def start(self):
        for gen in self.gens:
            p = Process(target=worker, args=(self.queue, gen))
            p.start()

            self.ps.append(p)

        return self.queue

    def stop(self):
        for p in self.ps:
            p.terminate()

        self.reset()

    @property
    def finished(self):
        return all([not p.is_alive() for p in self.ps])
