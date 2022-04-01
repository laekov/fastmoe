import os


def float_from_env(key, default=-1):
    if key in os.environ:
        return float(os.environ[key])
    return default


def switch_from_env(key, default=False):
    if key in os.environ:
        return os.environ[key] in ['1', 'ON']
    return default
