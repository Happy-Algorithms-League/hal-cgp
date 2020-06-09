import functools
import hashlib
import numpy as np
import os
import pickle

from typing import Any, Callable, Dict, List, Tuple, Type, Union

from .node import primitives_dict, Node


def __check_cache_consistency(fn: str, func: Callable[..., float]) -> None:
    if os.path.isfile(fn):
        with open(fn, "rb") as f:
            try:
                cursor: Dict[str, Any] = pickle.load(f)
            except EOFError:
                return  # no entry yet, so not possible to check

            cached_item: Dict[str, Any] = list(cursor.values())[0]
            return_value: float = func(*cached_item["args"], **cached_item["kwargs"])

            if not np.isclose(return_value, cached_item["return_value"]):
                raise RuntimeError(
                    "inconsistent return values"
                    " -- are different functions using the same cache file?"
                    " please use different cache files for different functions"
                )


def __compute_key_from_args(*args: Any, **kwargs: Any) -> str:
    s: str = str(args) + str(kwargs)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def __find_result_in_cache_file(fn: str, key: str) -> Union[float, None]:
    if os.path.isfile(fn):
        with open(fn, "rb") as f:
            while True:
                try:
                    cursor: Dict[str, Any] = pickle.load(f)
                except EOFError:
                    break

                if key in cursor:
                    return cursor[key]["return_value"]

    return None


def __store_new_cache_entry(
    fn: str, key: str, return_value: float, args: Tuple, kwargs: Dict[str, Any]
) -> None:
    with open(fn, "ab") as f:

        result = {
            "args": args,
            "kwargs": kwargs,
            "return_value": return_value,
        }

        pickle.dump({key: result}, f)


def disk_cache(fn: str) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Cache function return values on disk.

    Decorator that caches a functions return values on disk. Next time the
    decorated function is called with the same arguments it returns the stored
    values from disk instead of executing the function.

    Consistency of the cache is checked upon decorating the function
    by making sure the it returns the same value as the first
    argument/keyword argument combination found in the cache.

    WARNING: this implementation is neither optimized for speed nor storage
    space and does not limit the size of the cache file.

    WARNING: the consistency check may pass incorrectly if the
    decorated function happens to return a consistent value for the
    first argument from the cache although it returns different values
    for other arguments.

    WARNING: avoid using the decorator on nested functions as the
    consistency check will be applied on each decoration thus doubling
    the runtime.

    Parameters
    ----------
    fn : Callable
        Function to be cached.

    """

    def decorator(func: Callable[..., float]) -> Callable[..., float]:
        __check_cache_consistency(fn, func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Union[float, None]:

            key: str = __compute_key_from_args(*args, **kwargs)

            result_value_cached: Union[float, None] = __find_result_in_cache_file(fn, key)
            if result_value_cached is not None:
                return result_value_cached

            return_value: float = func(*args, **kwargs)
            __store_new_cache_entry(fn, key, return_value, args, kwargs)

            return return_value

        return wrapper

    return decorator


def primitives_from_class_names(primitives_str: Tuple[str, ...]) -> Tuple[Type[Node], ...]:

    primitives: List[Type[Node]] = []
    for s in primitives_str:
        primitives.append(primitives_dict[s])

    return tuple(primitives)
