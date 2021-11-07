from itertools import chain

import numpy as np
import pandas as pd
from tqdm import tqdm

from batch_generator.batch_generator import BatchGenerator
from batch_generator.dir import DirIterator


class Tester:
    def __init__(self, mc, test_data_dir, flavor, model_box, metrics):
        self.mc = mc

        sess = self.mc.read_model()

        iterator = DirIterator.from_data_folder(test_data_dir)
        gen = BatchGenerator(iterator, self.mc, flavor, False)

        ops = [model_box.context_tower, model_box.reply_tower] + metrics

        infos = []
        embeddings = []
        metrics_vals = []

        for info, batch in tqdm(gen):
            output = model_box.run(sess, ops, batch)

            infos.extend(chain.from_iterable(info))
            embeddings.extend(chain.from_iterable(zip(*output[:2])))
            metrics_vals.append(output[2:])

        infos = pd.DataFrame(infos, columns=['message'])
        infos['type'] = pd.Series(infos.index % 2).map({0: 'context', 1: 'reply'})

        self.data = {
            'embeddings': np.array(embeddings),
            'metrics': np.array(metrics_vals),
            'infos': infos
        }

    def get_score(self):
        return self.data['metrics'].mean(axis=0)

    def write_embeddings(self, size):
        embeddings = self.data['embeddings']
        inds = np.random.choice(embeddings.shape[0], size, replace=False)
        embeddings = embeddings[inds]

        meta = self.data['infos'].loc[inds]

        self.mc.write_embeddings(meta, embeddings)
