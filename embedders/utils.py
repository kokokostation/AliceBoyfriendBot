DEVICE_MAPPING = {
    'cpu': "/device:CPU:0",
    'gpu': "/device:GPU:0"
}


def get_device(mp):
    return DEVICE_MAPPING[mp.get('device', 'cpu')]