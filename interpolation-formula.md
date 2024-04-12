---
Original author: akalenuk
Link: https://wordsandbuttons.online/binary_search.html
---

# Linear Interpolation Formula

Using linear interpolation for range subdivisions in uniform distributions during binary search is very efficient.

Formula is:

```
i_d = (i1(A[i2] - x) + i2(x - A[i1])) / (A[i2] - A[i1])
i_d = dividing index, 
i1 = start index of sub-range,
i2 = stop index of sub-range
x = element to search for,
A = array to search in

index_div = (index_1(Array[index_2] - x) + index_2(x - Array[index_1])) / (Array[index_2] - Array[index_1])

```
