import numpy as np

from trainer.utils import get_log


def get_last_val_summary(trainer, summary_name):
    val_log = trainer.model_container.dirs['val']
    log = get_log(val_log, summary_name)
    vals = np.array(log['vals'])

    return vals[-trainer.epoch_steps // trainer.val_steps:].mean()


def get_last_val_loss(trainer):
    return get_last_val_summary(trainer, trainer.loss[0])
