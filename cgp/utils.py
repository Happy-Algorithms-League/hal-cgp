import functools
import hashlib
import os
import pickle
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy as np

from .individual import IndividualMultiGenome, IndividualSingleGenome
from .node import Node, primitives_dict


def __check_cache_consistency(fn: str, func: Callable[..., float]) -> None:
    """Retrieve an entry from the cache, execute the callable with the
    cached arguments and check whether the return value matches the
    cached result.

    WARNING: consistency is only checked when a _finite_ return value
    can be found in the cache.

    """
    cached_item: Union[Dict[str, Any], None] = __find_item_with_finite_return_value(fn)
    if cached_item is None:
        return

    return_value: float = func(*cached_item["args"], **cached_item["kwargs"])

    if not np.isclose(return_value, cached_item["return_value"]):
        raise RuntimeError(
            "inconsistent return values"
            " -- are different functions using the same cache file?"
            " please use different cache files for different functions"
        )


def __find_item_with_finite_return_value(fn: str) -> Union[Dict[str, Any], None]:
    """Find a cache entry which has a finite return value."""
    if os.path.isfile(fn):
        with open(fn, "rb") as f:
            while True:
                try:
                    cursor: Dict[str, Any] = pickle.load(f)
                except EOFError:
                    return None  # no entry yet, so not possible to check

                cached_item: Dict[str, Any] = list(cursor.values())[0]
                if np.isfinite(cached_item["return_value"]):
                    return cached_item
    else:
        return None


def __compute_key_from_args(*args: Any, **kwargs: Any) -> str:
    """Compute a key from the arguments passed to the decorated
    function.

    """

    s: str = str(args) + str(kwargs)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def __compute_key_from_evaluation(
    seed: int, min_value: float, max_value: float, batch_size: int, *args: Any
) -> str:
    """Compute a key for the function encoded in an individual by
    evaluating it's NumPy expression on random input samples and
    hashing the output values.

    """

    if not (
        isinstance(args[0], IndividualSingleGenome) or isinstance(args[0], IndividualMultiGenome)
    ):
        raise ValueError("first argument of decorated function must be an Individual instance")

    rng = np.random.RandomState(seed=seed)
    ind = args[0]
    if isinstance(ind, IndividualSingleGenome):
        f_single = ind.to_numpy()
        x = rng.uniform(min_value, max_value, (batch_size, ind.genome._n_inputs))
        y = f_single(x)
        s = np.array_str(y, precision=15)
    elif isinstance(ind, IndividualMultiGenome):
        f_multi = ind.to_numpy()
        s = ""
        for i in range(len(ind.genome)):
            x = rng.uniform(min_value, max_value, (batch_size, ind.genome[i]._n_inputs))
            y = f_multi[i](x)
            s += np.array_str(y, precision=15)
    else:
        assert False  # should never be reached

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

        result = {"args": args, "kwargs": kwargs, "return_value": return_value}

        pickle.dump({key: result}, f)


def disk_cache(
    fn: str,
    *,
    use_fec: bool = False,
    fec_seed: int = 0,
    fec_min_value: float = -100.0,
    fec_max_value: float = 100.0,
    fec_batch_size: int = 10
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Cache function return values on disk.

    Decorator that caches a function's return values on disk. Next time the
    decorated function is called with the same arguments it returns the stored
    values from disk instead of executing the function.

    Consistency of the cache is checked upon decorating the function
    by making sure the it returns the same value as the first
    argument/keyword argument combination found in the cache.

    If `use_fec` is `False` (default) the arguments of the decorated
    function are used to compute a hash. If `use_fec` is `True` the
    decorator uses functional equivalance checking [Real et al.,
    2020]: It generates a NumPy-compatible expression from the
    function's first argument (*must* be an `IndividualSingleGenome`
    or `IndividualMultiGenome` instance) and evaluates it on randomly
    generated values. The output values are then used to compute a
    hash.

    WARNING: this implementation is neither optimized for speed nor storage
    space and does not limit the size of the cache file.

    WARNING: the consistency check may pass incorrectly if the
    decorated function happens to return a consistent value for the
    first argument from the cache although it returns different values
    for other arguments.

    WARNING: avoid using the decorator on nested functions as the
    consistency check will be applied on each decoration thus doubling
    the runtime.

    References
    ----------
    Real, E., Liang, C., So, D. R., & Le, Q. V. (2020). AutoML-Zero:
    Evolving machine learning algorithms from scratch. arXiv preprint
    arXiv:2003.03384.

    Parameters
    ----------
    fn : str
        Name of the cache file.
    use_fec : bool
        Whether to use functional equivalance checking.
    fec_seed : int
        Seed value for fec.
    fec_min_value : float
        Minimal value for fec input samples.
    fec_max_value : float
        Maximal value for fec input samples.
    fec_batch_size : int
        Number of fec input samples.

    Returns
    -------
    Callable
        The decorated function.

    """

    def decorator(func: Callable[..., float]) -> Callable[..., float]:
        __check_cache_consistency(fn, func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Union[float, None]:

            key: str
            if use_fec:
                key = __compute_key_from_evaluation(
                    fec_seed, fec_min_value, fec_max_value, fec_batch_size, *args
                )
            else:
                key = __compute_key_from_args(*args, **kwargs)

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
