from trainer.interfaces.callback import Callback
from trainer.callbacks.utils import get_last_val_summary


class ReducePercentage(Callback):
    def __init__(self, percentage, threshold=0.7):
        self.percentage = percentage
        self.threshold = threshold

    def on_epoch_end(self, trainer):
        pao = get_last_val_summary(trainer, 'pao')

        if pao > self.threshold:
            self.percentage.set_value(trainer, max(0, self.percentage.value - 0.1))
