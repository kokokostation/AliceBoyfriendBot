import pandas as pd
import os
from functools import partial
from shutil import move
import numpy as np

from dataset.processing_utils import filter_and_compress
from batch_generator.prior import get_hash_set, make_hash_of_entry, CNTR_LOC
from batch_generator.dir import DirIterator
from batch_generator.batch_generator import BatchGenerator
from application.model import ApplicationModel
from dataset.utils import PackMaker
from train_utils.parallel_launcher import parallel_launcher_once
from application.faiss_storage import FaissStorage
from train_utils.utils import write_json
from batch_generator.utils import plain_batch_generator, transpose
from model.lstm_embedder import LSTMModel
from batch_generator.flavors import RECURRENT_FUSED_REPLY_FOR_RANKING, \
    RECURRENT_FUSED_RANKING_PREPARATION
from train_utils.container import ModelContainer
from application.utils import make_replies
from batch_generator.prior import make_priors, REPLIES_DIR, CNTR_LOC
from train_utils.utils import get_files


def hash_filter(hash_set, entry):
    return make_hash_of_entry(entry) in hash_set


def filter_frequent_replies(data_dir, output_dir, prior_dir, limit=2000000):
    hash_set = get_hash_set(pd.read_pickle(os.path.join(prior_dir, CNTR_LOC)), limit)

    filter_and_compress(data_dir, partial(hash_filter, hash_set), output_dir)


def make_context_worker(files, data, index):
    output_dir, mc, model, flavor, pack_size = data

    iterator = DirIterator(files)
    gen = BatchGenerator(iterator, mc, flavor, False)
    app_model = ApplicationModel(mc, model, flavor)
    portion = []
    pack_maker = PackMaker(pack_size, output_dir, index, pd.to_pickle)

    for full, batch in gen:
        portion.extend(list(zip(full, app_model.context_embedding_batch(batch))))

        pack_maker.make_pack(portion)

    pack_maker.finalize(portion)


def make_contexts(data_dir, output_dir, mc, model, flavor, pack_size, pool_size):
    parallel_launcher_once(data_dir, (output_dir, mc, model, flavor, pack_size),
                           make_context_worker, pool_size)


def faiss_worker(files, data, index):
    multiplier, cuda_devices, replies_data = data

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_devices[index])

    replies = pd.read_pickle(replies_data)
    vectors = replies['vectors']['vectors']
    storage = FaissStorage(vectors.shape[1], multiplier)
    storage.fit(vectors)

    del replies['vectors']

    for file in files:
        result = []

        for entries, context_embs in plain_batch_generator(
                pd.read_pickle(file), 500, transpose):
            for entry, reply_inds in zip(entries, storage.predict(np.array(context_embs))):
                context, reply = entry[:-1], entry[-1]

                best = [replies['replies'][i] for i in reply_inds]
                if reply not in best:
                    best[0] = reply
                else:
                    index = best.index(reply)
                    best[0], best[index] = best[index], best[0]

                result.append([context, best])

        write_json(result, file)


def make_competitors(data_dir, multiplier, replies_data, cuda_devices):
    parallel_launcher_once(data_dir, (multiplier, cuda_devices, replies_data),
                           faiss_worker, len(cuda_devices))


def make_dataset_for_ranking(lstm_dir, independant_train_data, ranking_train_data,
                             reply_data, output_dir, cuda_devices):
    replies_data = os.path.join(reply_data, 'prior_replies.pickle')
    mc = ModelContainer(lstm_dir, ckpt='reddit_tf_weights', dill=True)
    model = LSTMModel(mc, False)

    make_priors(independant_train_data, reply_data)

    tm = ApplicationModel(mc, model=model, flavor=RECURRENT_FUSED_REPLY_FOR_RANKING(mc))

    iterator = DirIterator.from_data_folder(os.path.join(reply_data, REPLIES_DIR))
    gen = BatchGenerator(iterator, mc, RECURRENT_FUSED_REPLY_FOR_RANKING(mc), infinite=False)

    pd.to_pickle(make_replies(tm, gen), replies_data)

    for i, files in enumerate(plain_batch_generator(get_files(ranking_train_data), 50)):
        curr_output = os.path.join(output_dir, 'folder_{}'.format(i))

        os.makedirs(curr_output)

        make_contexts(files, curr_output, mc, model, RECURRENT_FUSED_RANKING_PREPARATION(mc),
                      100000, 10)

        make_competitors(curr_output, 10, replies_data, cuda_devices)

    fnum = 0
    folders = get_files(output_dir, folders_ok=True)
    for folder in folders:
        for file in get_files(folder):
            move(file, os.path.join(output_dir, str(fnum)))
            fnum += 1

    for folder in folders:
        os.rmdir(folder)
