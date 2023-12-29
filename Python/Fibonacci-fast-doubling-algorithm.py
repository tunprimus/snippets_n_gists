# 
# Fast doubling Fibonacci algorithm (Python)
# by Project Nayuki, 2015. Public domain.
# https://www.nayuki.io/page/fast-fibonacci-algorithms
# 

def fibonacci(n):
    if n < 0:
        raise ValueError('Negative arguments not implemented')
    return _fib(n)[0]

def _fib(n):
    if n == 0:
        return (0, 1)
    else:
        a, b = _fib(n // 2)
        c = a * (b * 2 - a)
        d = a * a + b * b
        if n % 2 == 0:
            return (c, d)
        else:
            return (d, c + d)

print(fibonacci(10))
print(fibonacci(100))
