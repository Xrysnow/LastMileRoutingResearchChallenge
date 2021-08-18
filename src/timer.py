import time

class Timer:
    def __init__(self):
        self._t = time.perf_counter()

    def tic(self, log=True):
        cur = time.perf_counter()
        dt = cur - self._t
        self._t = cur
        if log:
            print('tic: %.3fs' % dt)
        return dt
