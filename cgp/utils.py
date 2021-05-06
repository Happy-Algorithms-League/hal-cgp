import functools
import hashlib
import os
import pickle
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from .individual import IndividualMultiGenome, IndividualSingleGenome
from .node import Node, primitives_dict

if TYPE_CHECKING:
    import multiprocessing as mp  # noqa: F401


def __check_cache_consistency(fn: str, func: Callable[..., float]) -> None:
    """Retrieve an entry from the cache, execute the callable with the
    cached arguments and check whether the return value matches the
    cached result.

    WARNING: consistency is only checked when a _finite_ return value
    can be found in the cache.

    """
    cached_item: Optional[Dict[str, Any]] = __find_args_and_return_value_for_consistency_check(fn)
    if cached_item is None:
        return

    return_value: float = func(*cached_item["args"], **cached_item["kwargs"])

    if not np.isclose(return_value, cached_item["return_value"]):
        raise RuntimeError(
            "inconsistent return values"
            " -- are different functions using the same cache file?"
            " please use different cache files for different functions"
        )


def __find_args_and_return_value_for_consistency_check(fn: str) -> Optional[Dict[str, Any]]:
    """Try to retrieve argument and return value for consistency check."""
    if os.path.isfile(fn):
        with open(fn, "rb") as f:
            try:
                res: Dict[str, Any] = pickle.load(f)
            except EOFError:
                return None

        if "args_return_value_consistency_check" in res and np.all(
            np.isfinite(res["args_return_value_consistency_check"]["return_value"])
        ):
            return res["args_return_value_consistency_check"]

    return None


def compute_key_from_sympy_expr_and_args(*args: Any, **kwargs: Any) -> str:
    """Compute a key from the sympy expression encoded in an individual
    and the remaining arguments passed to the decorated function.

    """

    if not (
        isinstance(args[0], IndividualSingleGenome) or isinstance(args[0], IndividualMultiGenome)
    ):
        raise ValueError("first argument of decorated function must be an Individual instance")

    s: str = str(args[0].to_sympy()) + str(args[1:]) + str(kwargs)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def compute_key_from_numpy_evaluation_and_args(
    *args: Any,
    _seed: int = 0,
    _min_value: float = -100.0,
    _max_value: float = 100.0,
    _batch_size: int = 10,
    **kwargs: Any,
) -> str:
    """Compute a key from the function encoded in an individual
    and the remaining arguments passed to the decorated function.

    The function is evaluated on random inputs and a key is generated
    by hashing the corresponding output values.

    Parameters
    ----------
    _seed : int, optional
        Seed value for fec. Defaults to 0.
    _min_value : float, optional
        Minimal value for fec input samples. Defaults to -100.0.
    _max_value : float, optional
        Maximal value for fec input samples. Defaults to 100.0.
    _batch_size : int, optional
        Number of fec input samples. Defaults to 10.

    """

    if not (
        isinstance(args[0], IndividualSingleGenome) or isinstance(args[0], IndividualMultiGenome)
    ):
        raise ValueError("first argument of decorated function must be an Individual instance")

    rng = np.random.RandomState(seed=_seed)
    ind = args[0]
    if isinstance(ind, IndividualSingleGenome):
        f_single = ind.to_numpy()
        x = [
            rng.uniform(_min_value, _max_value, size=_batch_size)
            for _ in range(ind.genome._n_inputs)
        ]
        y = f_single(*x)
        s = np.array_str(np.array(y), precision=15)
    elif isinstance(ind, IndividualMultiGenome):
        f_multi = ind.to_numpy()
        s = ""
        for i in range(len(ind.genome)):
            x = [
                rng.uniform(_min_value, _max_value, size=_batch_size)
                for _ in range(ind.genome[i]._n_inputs)
            ]
            y = f_multi[i](*x)
            s += np.array_str(np.array(y), precision=15)
    else:
        assert False  # should never be reached

    s += str(args[1:]) + str(kwargs)

    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def __find_result_in_cache_file(fn: str, key: str) -> Optional[float]:
    if os.path.isfile(fn):
        with open(fn, "rb") as f:

            try:
                res = pickle.load(f)
            except EOFError:
                return None

        if key in res:
            return res[key]
        else:
            return None

    return None


def __store_new_cache_entry(
    fn: str, key: str, return_value: float, args: Tuple, kwargs: Dict[str, Any]
) -> None:

    res: Dict[str, Any]
    try:
        with open(fn, "rb") as f:
            res = pickle.load(f)
    except (EOFError, FileNotFoundError):
        res = {}

    res[key] = return_value

    if "args_return_value_consistency_check" not in res or not np.all(
        np.isfinite(res["args_return_value_consistency_check"]["return_value"])
    ):
        res["args_return_value_consistency_check"] = {
            "args": args,
            "kwargs": kwargs,
            "return_value": return_value,
        }

    with open(fn, "wb") as f:
        pickle.dump(res, f)


def disk_cache(
    fn: str,
    *,
    compute_key: Callable[..., str] = compute_key_from_numpy_evaluation_and_args,
    file_lock: Optional["mp.synchronize.Lock"] = None,
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Cache function return values on disk.

    Decorator that caches a function's return values on disk. Next
    time the decorated function is called with the same arguments it
    returns the stored values from disk instead of executing the
    function. The first argument of the decorated function *must* be
    an IndividualBase instance.

    Consistency of the cache is checked upon decorating the function
    by making sure the it returns the same value as the first
    argument/keyword argument combination found in the cache.

    The `compute_key` parameter is a function receiving the aguments
    and keyword arguments of the decorated function and must return a
    unique key. By default, the decorator uses functional equivalance
    checking [Real et al., 2020]: It generates a NumPy-compatible
    expression from the function's first argument (*must* be an
    `IndividualSingleGenome` or `IndividualMultiGenome` instance) and
    evaluates it on randomly generated values. The output values are
    then used to compute a hash.

    WARNING: this implementation is neither optimized for speed nor storage
    space and does not limit the size of the cache file.

    WARNING: the consistency check may pass incorrectly if the
    decorated function happens to return a consistent value for the
    arguments from the cache although it returns different values for
    other arguments.

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
    compute_key : Callable[..., str], optional
        Function to compute a unique key from an individual and the
        remaining function arguments. Defaults to
        `compute_key_from_numpy_evaluation_and_args`.
    file_lock : multiprocessing.synchronize.Lock, optional
        Lock to make sure only a single process reads from/write to
        cache file. Defaults to None.

    Returns
    -------
    Callable
        The decorated function.

    """

    def decorator(func: Callable[..., float]) -> Callable[..., float]:
        __check_cache_consistency(fn, func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> float:

            key: str = compute_key(*args, **kwargs)

            if file_lock is not None:
                file_lock.acquire()

            result_value_cached: Optional[float] = __find_result_in_cache_file(fn, key)

            if file_lock is not None:
                file_lock.release()

            if result_value_cached is not None:
                return result_value_cached

            return_value: float = func(*args, **kwargs)

            if file_lock is not None:
                file_lock.acquire()

            __store_new_cache_entry(fn, key, return_value, args, kwargs)

            if file_lock is not None:
                file_lock.release()

            return return_value

        return wrapper

    return decorator


def primitives_from_class_names(primitives_str: Tuple[str, ...]) -> Tuple[Type[Node], ...]:

    primitives: List[Type[Node]] = []
    for s in primitives_str:
        primitives.append(primitives_dict[s])

    return tuple(primitives)
