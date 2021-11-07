from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_log(path, key):
    event_acc = EventAccumulator(path)
    event_acc.Reload()

    w_times, step_nums, vals = zip(*event_acc.Scalars(key))

    return {
        'w_times': w_times,
        'step_nums': step_nums,
        'vals': vals
    }
