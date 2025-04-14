#!/usr/bin/env python3
# Adapted from How to Effectively Time Code in Python Using Decorators and Timeit: -> https://sqlpey.com/python/timing-code-in-python/
from functools import wraps
from time import time

def timing_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_time_ns = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        end_time_ns = time.perf_counter_ns()
        
        execution_time = end_time - start_time
        execution_time_ns = end_time_ns - start_time_ns
        
        print(f"Function: {func.__name__} | Args: {args} | Took: {execution_time:.3f} seconds")
        print(f"Function: {func.__name__} | Args: {args} | Took: {execution_time_ns} ns ({(execution_time_ns / 1000000000):.3f} s)")
        
        return result
    return wrapper

