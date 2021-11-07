import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output

from train_utils.queue import MultiprocessQueue
from train_utils.utils import make_feed_dict


class Trainer:
    def __init__(self,
                 loss,
                 train_step,
                 metric,
                 train_gen,
                 val_gen,
                 sess,
                 placeholders,
                 report_step=2000,
                 val_steps=100,
                 alpha=0.99,
                 queue_jobs=2,
                 queue_maxsize=2000):

        self.loss = loss
        self.train_step = train_step
        self.metric = metric
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.val_steps = val_steps
        self.sess = sess
        self.report_step = report_step
        self.alpha = alpha
        self.placeholders = placeholders
        self.queue_jobs = queue_jobs
        self.queue_maxsize = queue_maxsize

        self.cur_step = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_metric_history = []
        self.val_metric_history = []
        self.train_loss = None
        self.train_metric = None
        self.val_loss = None
        self.val_metric = None

    def plot_results(self):
        steps = np.arange(0, self.cur_step + 1, self.report_step)
        clear_output(wait=True)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(steps, self.train_loss_history, label="train")
        plt.plot(steps, self.val_loss_history, label="validation")
        plt.legend(loc='upper right')
        plt.subplot(1, 2, 2)
        plt.title("Metric")
        plt.plot(steps, self.train_metric_history, label="train")
        plt.plot(steps, self.val_metric_history, label="validation")
        plt.legend(loc='lower right')
        plt.show()

    def smooth_update(self, acc_val, new_val):
        if acc_val is None:
            return new_val
        return self.alpha * acc_val + (1 - self.alpha) * new_val

    def make_feed_dict(self, batch):
        return make_feed_dict(self.placeholders, batch)

    def make_step(self, batch, train=True):
        if train:
            l, m, _ = self.sess.run([self.loss, self.metric, self.train_step],
                                    self.make_feed_dict(batch))
            self.train_loss = self.smooth_update(self.train_loss, l)
            self.train_metric = self.smooth_update(self.train_metric, m)

        else:
            l, m = self.sess.run([self.loss, self.metric],
                                 self.make_feed_dict(batch))
            self.val_loss = self.smooth_update(self.val_loss, l)
            self.val_metric = self.smooth_update(self.val_metric, m)

    def report_results(self, train=True):
        if train:
            self.train_loss_history.append(self.train_loss)
            self.train_metric_history.append(self.train_metric)
            self.train_loss = None
            self.train_metric = None
        else:
            self.val_loss_history.append(self.val_loss)
            self.val_metric_history.append(self.val_metric)
            self.val_loss = None
            self.val_metric = None

    def start(self, include_validation=True):
        queue = MultiprocessQueue.from_gen_maker(self.train_gen,
                                                 self.queue_jobs,
                                                 self.queue_maxsize)
        val_gen = iter(self.val_gen())

        try:
            q = queue.start()

            while True:
                batch = q.get()

                self.make_step(batch)
                if self.cur_step % self.report_step == 0:
                    self.report_results()

                    if include_validation:
                        for _ in range(self.val_steps):
                            val_batch = next(val_gen)
                            self.make_step(val_batch, train=False)
                        self.report_results(train=False)
                        self.plot_results()

                self.cur_step += 1
        finally:
            queue.stop()

    def save(self, weights_path):
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, weights_path)
