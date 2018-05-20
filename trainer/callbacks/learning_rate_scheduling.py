import numpy as np

from trainer.interfaces.callback import Callback


class LearnigRateSchedule(Callback):
    '''
        min_lr : float
            lower bound for learning rate
        max_lr : float
            upper bound for learning rate
        step_size : float
            cycle size
        decay : float
            learning rate decay coefficient on every step
        mode : string
            'triangular' or 'cosine'
    For additional info see:
        https://www.jeremyjordan.me/nn-learning-rate/
    '''
    def __init__(self, learning_rate, min_lr=0.0001, max_lr=0.001, step_size=4, decay=0.99, mode='triangular'):
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.decay = decay
        self.mode = self.__mode_checker(mode)

    @staticmethod
    def __mode_checker(mode):
        if mode in ['triangular', 'cosine']:
            return mode
        else:
            return 'triangular'

    def __get_cycle(self, iteration):
        return np.floor((1+iteration) / (2 * self.step_size))

    def __get_x(self, iteration):
        cycle = self.__get_cycle(iteration)
        return np.abs(iteration / self.step_size - 2 * cycle + 1)

    def triangular_lr(self, iteration):
        return self.min_lr + 0.5 * ((self.max_lr - self.min_lr) * self.__get_x(iteration)) * self.decay ** iteration

    def cosine_lr(self, iteration):
        return self.min_lr + 0.5 * ((self.max_lr - self.min_lr) * (1 + np.cos(iteration / self.step_size * np.pi))) * self.decay ** iteration

    def on_epoch_end(self, trainer):
        iteration = trainer.step
        iteration /= 500
        if self.mode == 'triangular':
            new_lr = self.triangular_lr(iteration)
        else:
            new_lr = self.cosine_lr(iteration)
        trainer.sess.run(self.learning_rate.assign(new_lr))
