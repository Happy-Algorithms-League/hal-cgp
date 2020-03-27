import functools
import hashlib
import os
import pickle

from .node import primitives_dict


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

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            # compute hash from function arguments
            s = str(args) + str(kwargs)
            key = hashlib.sha1(s.encode("utf-8")).hexdigest()

            # check whether result exists in cache file
            if os.path.isfile(fn):
                with open(fn, "rb") as f:
                    # iterate over all stored pickle streams
                    while True:
                        try:
                            cursor = pickle.load(f)
                        except EOFError:
                            break

                        if key in cursor:
                            return cursor[key]  # result was found

            # if result does not exist, compute return values and store new entry
            # in cache file
            return_values = func(*args, **kwargs)

            with open(fn, "ab") as f:  # append new pickle stream
                pickle.dump({key: return_values}, f)

            return return_values

        return wrapper

    return decorator


def primitives_from_class_names(primitives_str):

    primitives = []
    for s in primitives_str:
        primitives.append(primitives_dict[s])

    return primitives
