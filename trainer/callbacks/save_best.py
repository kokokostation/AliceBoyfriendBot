import numpy as np

from trainer.callbacks.utils import get_last_val_loss
from trainer.interfaces.callback import Callback


class SaveBest(Callback):
    def __init__(self, patience, one_ckpt=True):
        self.patience = patience
        self.one_ckpt = one_ckpt
        self.wait = 0
        self.best = np.inf

    def on_epoch_end(self, trainer):
        self.wait += 1

        last = get_last_val_loss(trainer)

        if last < self.best:
            self.best = last

            if self.wait >= self.patience:
                trainer.model_container.save_model(trainer.sess,
                                                   trainer.step if not self.one_ckpt else -1)

                self.wait = 0
