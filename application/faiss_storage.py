from faiss import IndexFlatIP, StandardGpuResources, index_cpu_to_gpu
import numpy as np

from application.interfaces.vector_storage import VectorStorage


class FaissStorage(VectorStorage):
    def __init__(self, d, k=1):
        self.index = IndexFlatIP(d)
        self.res = StandardGpuResources()
        self.index = index_cpu_to_gpu(self.res, 0, self.index)
        self.k = k

    def fit(self, vectors):
        self.index.add(vectors)

        return self

    def predict(self, vectors):
        D, I = self.index.search(vectors, self.k)

        return I
