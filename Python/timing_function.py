#!/usr/bin/env python3
# Adapted from How to Effectively Time Code in Python Using Decorators and Timeit: -> https://sqlpey.com/python/timing-code-in-python/
from functools import wraps
from time import time

def timing_function(func, show_full_params=False):
    """
    A decorator that measures and prints the execution time of a function.

    Parameters
    ----------
    func : callable
        The function to be timed.
    show_full_params : bool, optional
        If True, prints the function name and arguments along with the timing information. Defaults to False.

    Returns
    -------
    callable
        A wrapped version of the input function that prints its execution time.

    Notes
    -----
    The execution time is measured in both seconds and nanoseconds.
    If `show_full_params` is True, the function name and arguments are also displayed.

    Example
    -------
    >>> from timing_function import timing_function
    >>>
    >>> @timing_function
    >>> def test_function():
    >>>     print("Hello World!" * 13)
    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_time_ns = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        end_time_ns = time.perf_counter_ns()

        execution_time = end_time - start_time
        execution_time_ns = end_time_ns - start_time_ns

        if show_full_params:
            print(f"Function: `{func.__name__}` | Args: {args} | Took: {execution_time:.3f} seconds")
            print(f"Function: `{func.__name__}` | Args: {args} | Took: {execution_time_ns}ns ({(execution_time_ns / 1000000000):.3f}s)")
        else:
            print(f"Function: `{func.__name__}` took --> {execution_time_ns}ns ({(execution_time_ns / 1000000000):.3f}s)")

        return result
    return wrapper

