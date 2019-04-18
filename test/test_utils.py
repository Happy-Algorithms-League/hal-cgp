import sys
import tempfile
import time

sys.path.insert(0, '../')
import gp


def test_cache_decorator():

    sleep_time = 0.1

    @gp.utils.disk_cache(tempfile.mkstemp()[1])
    def objective(label):
        time.sleep(sleep_time) # simulate long execution time
        return label

    # first call should take long due to sleep
    t0 = time.time()
    objective('test')
    assert time.time() - t0 > sleep_time / 2.

    # second call should be faster as result is retrieved from cache
    t0 = time.time()
    objective('test')
    assert time.time() - t0 < sleep_time / 2.
