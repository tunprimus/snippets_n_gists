#!/usr/bin/env python3
"""
Custom Multi-Variable List Iterator

Here's a succinct explanation of the `CustomListIterator` class:

-* `__init__(self, lst, num_var=1)`: Initialises the iterator with a list `lst` and an optional number of variables `num_var` (defaulting to 1). It sets the list, number of variables, and index to 0.
-* `__iter__(self)`: Returns the iterator object itself, allowing it to be used in a `for` loop.
-* `__next__(self)`: Returns the next `num_var` elements from the list as a tuple. If there are not enough elements left, it raises a `StopIteration` exception.

In summary, this class allows you to iterate over a list in chunks of a specified size, returning each chunk as a tuple.

For example:

test_list01 = [10, 20, 30, 40, 50]
for a in CustomListIterator(test_list01):
    print(f"a: {a}")

Results in:
a: (10,)
a: (20,)
a: (30,)
a: (40,)
a: (50,)

test_list02 = [12, 22, 32, 42, 52, 62, 72, 82]
for a, b in CustomListIterator(test_list02, 2):
    print(f"a: {a}, b: {b}")

Results in:
a: 12, b: 22
a: 32, b: 42
a: 52, b: 62
a: 72, b: 82

test_list03 = [13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123]
for a, b, c in CustomListIterator(test_list03, 3):
    print(f"a: {a}, b: {b}, c: {c}")

Results in:
a: 13, b: 23, c: 33
a: 43, b: 53, c: 63
a: 73, b: 83, c: 93
a: 103, b: 113, c: 123

test_list04 = [14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114, 124, 134, 144, 154, 164, 174, 184, 194, 204]
for a, b, c, d in CustomListIterator(test_list04, 4):
    print(f"a: {a}, b: {b}, c: {c}, d: {d}")

Results in:
a: 14, b: 24, c: 34, d: 44
a: 54, b: 64, c: 74, d: 84
a: 94, b: 104, c: 114, d: 124
a: 134, b: 144, c: 154, d: 164
a: 174, b: 184, c: 194, d: 204

"""
class CustomListIterator:
    def __init__(self, lst, num_var=1):
        """
        Initialises the iterator with a list `lst` and an optional number of variables `num_var` (defaulting to 1).
        It sets the list, number of variables, and index to 0.
        """
        self.lst = lst
        self.num_var = num_var
        self.index = 0
    
    def __iter__(self):
        """
        Returns the iterator object itself, allowing it to be used in a `for` loop.
        """
        return self
    
    def __next__(self):
        """
        Returns the next tuple of values from the list.

        The tuple will have length `num_var`, and will be padded with `None` values if there
        are not enough values left in the list. If the list has been fully iterated over,
        a `StopIteration` exception is raised.
        """
        buffer = self.lst[self.index:self.index+self.num_var]
        if len(buffer) == 0:
            raise StopIteration
        self.index += self.num_var
        return tuple((buffer + self.num_var * [None])[:self.num_var])
