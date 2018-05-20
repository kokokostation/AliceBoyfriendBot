from multiprocessing_on_dill import Pool
import numpy as np

from train_utils.utils import get_files


def modified_get_files(data_dir):
    if isinstance(data_dir, list):
        return data_dir
    else:
        return get_files(data_dir)


def parallel_launcher(data_dir, data, worker, pool_size, files_num):
    files = modified_get_files(data_dir)

    batches = [(files[i:i + files_num], data, j)
               for j, i in enumerate(range(0, len(files), files_num))]

    pool = Pool(pool_size)
    output = pool.starmap(worker, batches)

    pool.close()
    pool.join()

    return output


def parallel_launcher_once(data_dir, data, worker, pool_size):
    all_files_num = len(modified_get_files(data_dir))

    return parallel_launcher(data_dir, data, worker, pool_size,
                             int(np.ceil(all_files_num / pool_size)))