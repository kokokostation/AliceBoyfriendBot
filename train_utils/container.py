import tensorflow as tf
from functools import partial
import pandas as pd
import os
from tensorflow.contrib.framework import list_variables
import shutil
from tensorflow.contrib.tensorboard.plugins import projector

from train_utils.utils import write_json, read_dill, write_dill
from batch_generator.prior import CNTR_LOC


class ModelContainer:
    FNAMES = {
        'sparsifiers': 'sparsifiers.pickle',
        'counter': CNTR_LOC,
        'model_params': 'model_params.pickle',
        'applier': 'applier.pickle',
        'resource': 'resource.json'
    }
    READERS = {
        'json': partial(pd.read_json, typ='series'),
        'pickle': pd.read_pickle,
        'dill': read_dill
    }
    WRITERS = {
        'json': write_json,
        'pickle': pd.to_pickle,
        'dill': write_dill
    }
    DIRS = ['train', 'val', 'testing']

    def __init__(self, data_dir, ckpt, dill=False):
        self.data_dir = data_dir
        self.ckpt = os.path.join(data_dir, ckpt)
        self.dill = dill

        self.dirs = {key: self.dir_prefix(key) for key in ModelContainer.DIRS}

        for dir in self.dirs.values():
            os.makedirs(dir, exist_ok=True)

    def clear_logs(self):
        for dir in self.dirs.values():
            shutil.rmtree(dir)
            os.makedirs(dir)

    def ckpt_step(self, step):
        return '{}_{}'.format(self.ckpt, step)

    def read_model(self, step=-1):
        sess = tf.InteractiveSession()

        sess.run(tf.global_variables_initializer())

        ckpt = self.ckpt_step(step)

        if tf.train.checkpoint_exists(ckpt):
            names = set([a for a, _ in list_variables(ckpt)])
            vars = [v for v in tf.global_variables() if v.op.name in names]
            saver = tf.train.Saver(vars)
            saver.restore(sess, ckpt)

        return sess

    def save_model(self, sess, step=-1):
        saver = tf.train.Saver()
        saver.save(sess, self.ckpt_step(step))

    def dir_prefix(self, suffix):
        return os.path.join(self.data_dir, suffix)

    def dir_prefix_handle(self, handle):
        return self.dir_prefix(ModelContainer.FNAMES[handle])

    def get_path(self, handle):
        path = self.dir_prefix_handle(handle)
        ext = os.path.splitext(path)[-1][1:]

        if self.dill and ext == 'pickle':
            ext = 'dill'

        return path, ext

    def read(self, handle):
        path, ext = self.get_path(handle)
        return ModelContainer.READERS[ext](path)

    def write(self, handle, obj):
        path, ext = self.get_path(handle)

        ModelContainer.WRITERS[ext](obj, path)

    def exists(self, handle):
        return os.path.exists(self.dir_prefix_handle(handle))

    def write_embeddings(self, meta, embeddings):
        testing_dir = self.dirs['testing']

        meta_path = os.path.join(testing_dir, 'embeddings_meta.csv')
        meta.to_csv(meta_path, index=False, sep='\t')

        with tf.Graph().as_default():
            embedding_var = tf.Variable(embeddings,
                                        name='word_embeddings')

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            saver.save(sess, os.path.join(testing_dir, 'embeddings'))

            config = projector.ProjectorConfig()

            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            embedding.metadata_path = meta_path

            summary_writer = tf.summary.FileWriter(testing_dir)

            projector.visualize_embeddings(summary_writer, config)

    def model_params(self, train=True):
        mp = self.read('model_params')

        for value in mp.values():
            if isinstance(value, dict):
                value['train'] = train
        mp['train'] = train

        return mp