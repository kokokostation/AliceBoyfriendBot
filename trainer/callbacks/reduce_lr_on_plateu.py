import numpy as np

from trainer.callbacks.utils import get_last_val_loss
from trainer.interfaces.callback import Callback


class ReduceLROnPlateu(Callback):
    def __init__(self, learning_rate, factor, patience, min_lr):
        self.learning_rate = learning_rate
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.wait = 0
        self.best = np.inf

    def on_epoch_end(self, trainer):
        self.wait += 1

        last = get_last_val_loss(trainer)

        if last < self.best:
            self.best = last
            self.wait = 0
        else:
            if self.wait >= self.patience:
                old_lr = trainer.sess.run(self.learning_rate)

                if old_lr > self.min_lr:
                    new_lr = max(old_lr * self.factor, self.min_lr)

                    trainer.sess.run(self.learning_rate.assign(new_lr))

                    self.wait = 0
