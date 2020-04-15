import functools
import hashlib
import numpy as np
import os
import pickle

from .node import primitives_dict


def __check_cache_consistency(fn, func, args, kwargs):
    if os.path.isfile(fn):
        with open(fn, "rb") as f:
            try:
                cursor = pickle.load(f)
            except EOFError:
                return  # no entry yet, so not possible to check

            cached_item = list(cursor.values())[0]

            if len(args) != len(cached_item["args"]):
                raise RuntimeError(
                    "inconsistent arguments"
                    " -- are different functions using the same cache file?"
                    " please use different cache files for different functions"
                )

            if kwargs.keys() != cached_item["kwargs"].keys():
                raise RuntimeError(
                    "inconsistent keyword arguments"
                    " -- are different functions using the same cache file?"
                    " please use different cache files for different functions"
                )

            return_value = func(*cached_item["args"], **cached_item["kwargs"])

            if not np.isclose(return_value, cached_item["return_value"]):
                raise RuntimeError(
                    "inconsistent return values"
                    " -- are different functions using the same cache file?"
                    " please use different cache files for different functions"
                )


def __compute_key_from_args(*args, **kwargs):
    s = str(args) + str(kwargs)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def __find_result_in_cache_file(fn, key):
    if os.path.isfile(fn):
        with open(fn, "rb") as f:
            while True:
                try:
                    cursor = pickle.load(f)
                except EOFError:
                    break

                if key in cursor:
                    return cursor[key]

    return None


def __store_new_cache_entry(fn, key, return_value, args, kwargs):
    with open(fn, "ab") as f:

        result = {
            "args": args,
            "kwargs": kwargs,
            "return_value": return_value,
        }

        pickle.dump({key: result}, f)


def disk_cache(fn):
    """Cache function return values on disk.

    Decorator that caches a functions return values on disk. Next time the
    decorated function is called with the same arguments it returns the stored
    values from disk instead of executing the function.

    Warning: this implementation is neither optimized for speed nor storage
    space and does not limit the size of the cache file.

    Parameters
    ----------
    fn : Callable
        Function to be cached.
    """
    first_function_call = True

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            nonlocal first_function_call
            if first_function_call:
                __check_cache_consistency(fn, func, args, kwargs)
                first_function_call = False

            key = __compute_key_from_args(*args, **kwargs)

            result_cached = __find_result_in_cache_file(fn, key)
            if result_cached is not None:
                return result_cached["return_value"]

            return_value = func(*args, **kwargs)
            __store_new_cache_entry(fn, key, return_value, args, kwargs)

            return return_value

        return wrapper

    return decorator


def primitives_from_class_names(primitives_str):

    primitives = []
    for s in primitives_str:
        primitives.append(primitives_dict[s])

    return primitives
