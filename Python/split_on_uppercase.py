#!/usr/bin/env python3


def split_on_uppercase(target_str, keep_contiguous=False):
    """
    Splits a given string into parts based on uppercase letters and returns them as a list.
    Adapted from: https://stackoverflow.com/a/40382663

    Args:
        target_str (str): The input string to be split.
        keep_contiguous (bool, optional): If True, splits on contiguous uppercase letters. Defaults to False.

    Returns:
        list: A list of substrings split from the input string.
    Example:
    ----------
    >>> split_on_uppercase("theLongWindingRoad")
    ['the', 'Long', 'Winding', 'Road']
    >>> split_on_uppercase("TheLongWindingRoad")
    ['The', 'Long', 'Winding', 'Road']
    >>> split_on_uppercase("TheLongWINDINGRoad")
    ['The', 'Long', 'WINDING', 'Road']
    >>> split_on_uppercase("The123Long456Wind789ingRoad")
    ['The123', 'Long456', 'Wind789ing', 'Road']
    >>> split_on_uppercase("The123Long456WindingRoad789")
    ['The123', 'Long456', 'Winding', 'Road789']
    >>> split_on_uppercase("ABC")
    ['A', 'B', 'C']
    >>> split_on_uppercase("ABCD", True)
    ['ABCD']
    >>> split_on_uppercase(" ")
    [' ']
    >>> split_on_uppercase("")
    ['']
    """
    string_length = len(target_str)
    is_lower_around = (
        lambda: target_str[i - 1].islower()
        or string_length > (i + 1)
        and target_str[i + 1].islower()
    )
    start = 0
    parts = []
    for i in range(1, string_length):
        if target_str[i].isupper() and (not keep_contiguous or is_lower_around()):
            parts.append(target_str[start:i])
            start = i
    parts.append(target_str[start:])
    return parts
