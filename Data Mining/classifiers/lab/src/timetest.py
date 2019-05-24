import time


def timeit(func):
    """
    装饰器，计算函数执行时间@timeit
    """
    def wrapper(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        exec_time = time_end - time_start
        print(f"{func.__name__} exec time: {exec_time}s")
        return result
    return wrapper