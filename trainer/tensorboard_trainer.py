from itertools import count
import tensorflow as tf

from train_utils.queue import MultiprocessQueue
from trainer.utils import get_log


class TensorBoardTrainer:
    def __init__(self, model_container, train_step, loss, metrics, model_box,
                 train_gen, val_gen, callbacks=None,
                 epoch_steps=2000, val_steps=100,
                 queue_jobs=2, queue_maxsize=200, model_step=-1):
        self.model_container = model_container

        self.train_step = train_step
        self.loss = loss
        self.metrics = metrics
        self.model_box = model_box

        try:
            log = get_log(self.model_container.dirs['train'], self.loss[0])

            self.step = log['step_nums'][-1]
        except KeyError:
            self.step = 0

        for name, tensor in [self.loss] + self.metrics:
            tf.summary.scalar(name, tensor)
        self.merged = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(self.model_container.dirs['train'])
        self.val_writer = tf.summary.FileWriter(self.model_container.dirs['val'])

        self.train_gen = train_gen
        self.val_gen = val_gen

        self.callbacks = [] if callbacks is None else callbacks

        self.epoch_steps = epoch_steps
        self.val_steps = val_steps

        self.queue_jobs = queue_jobs
        self.queue_maxsize = queue_maxsize

        self.sess = model_container.read_model(model_step)

    def run(self, batch, train=True):
        ops = [self.merged]
        if train:
            ops.append(self.train_step)

        return self.model_box.run(self.sess, ops, batch)[0]

    def start(self):
        queue = MultiprocessQueue.from_gen_maker(self.train_gen,
                                                 self.queue_jobs,
                                                 self.queue_maxsize)
        val_gen = iter(self.val_gen())

        try:
            q = queue.start()

            for self.step in count(self.step + 1):
                _, batch = q.get()

                summary = self.run(batch)
                self.train_writer.add_summary(summary, self.step)

                if self.step % self.epoch_steps == 0:
                    for cb in self.callbacks:
                        cb.on_epoch_end(self)

                if self.step % self.val_steps == 0:
                    _, val_batch = next(val_gen)
                    summary = self.run(val_batch, train=False)

                    self.val_writer.add_summary(summary, self.step)
        finally:
            queue.stop()

    def save(self, step=-1):
        self.model_container.save_model(self.sess, step)

    def clear_logs(self):
        self.model_container.clear_logs()
        self.step = 0
