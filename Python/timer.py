#!/usr/bin/env python3
# Adapted from realpython/materials: -> https://github.com/realpython/materials/blob/master/pandas-fast-flexible-intuitive/tutorial/timer.py
import sys

def timeit_decorator(_func=None, *, repeat=3, number=1000, file=sys.stdout):
    """
    A decorator to measure and print time from best of `repeat` trials taken by a function.
    Mimics `timeit.repeat()`, but avg. time is printed.
    Returns function result and prints time.

    Parameters
    ----------
    _func : callable, optional
        The function to be timed. If not provided, this function can be used as a decorator.
    repeat : int, optional
        The number of times to repeat the timing. Defaults to 3.
    number : int, optional
        The number of times to call the function in each repeat. Defaults to 1000.
    file : file-like, optional
        The file to write the results to. Defaults to sys.stdout.

    Returns
    -------
    A callable that wraps the original function and prints the timing results to the specified file.

    Examples
    --------
    from .timer import timeit_decorator

    >>> @timeit_decorator
    ... def f():
    ...     return "-".join(str(n) for n in range(100))
    >>> @timeit_decorator(number=100000)
    ... def g():
    ...     return "-".join(str(n) for n in range(10))
    """
    import functools
    import gc
    import itertools
    import sys
    from timeit import default_timer as _timer

    _repeat = functools.partial(itertools.repeat, None)

    def wrap(func):
        @functools.wraps(func)
        def _timeit_decorator(*args, **kwargs):
            # Temporarily turn off garbage collection during the timing.
            # Makes independent timings more comparable.
            # If it was originally enabled, switch it back on afterwards.
            gcold = gc.isenabled()
            gc.disable()

            try:
                # Outer loop for the number of repetitions
                trials = []
                for _ in _repeat(repeat):
                    # Inner loop for the number of calls within each repeat
                    total = 0
                    for _ in _repeat(number):
                        start = _timer()
                        result = func(*args, **kwargs)
                        end = _timer()
                        total += end - start
                    trials.append(total)
                # We want the *average time* from the *best* trial.
                # For more on this methodology, see the docs for
                # Python's `timeit` module.
                #
                # "In a typical case, the lowest value gives a lower bound
                # for how fast your machine can run the given code snippet;
                # higher values in the result vector are typically not
                # caused by variability in Python’s speed, but by other
                # processes interfering with your timing accuracy."
                best = min(trials) / number
                # Calculate the standard deviation of the trials
                std = (sum((trial - best) ** 2 for trial in trials) / len(trials)) ** 0.5
                print(f"Best of {repeat} with {number} function calls per trial:")
                print(f"Function {func.__name__} ran in average of {best:.3f} seconds with standard deviation of {std:.3f}", end="\n\n", file=file)
            finally:
                # Switch garbage collection back on if it was originally on
                if gcold:
                    gc.enable()
            # Result is returned *only once*
            return result

        return _timeit_decorator

    # Syntax trick from Python @dataclass
    if _func is None:
        return wrap
    else:
        return wrap(_func)
