//@ts-check

/**
 * Calculate Fibonacci number with fast doubling algorithm
 * @author Project Nayuki, 2023. Public domain.
 * @link https://www.nayuki.io/page/fast-fibonacci-algorithms
 * @param {number} n The Fibonacci number to calculate
 * @returns {object} An array object
 */
function fibonacci(n) {
	if (n < 0) {
		throw RangeError('Negative arguments not implemented');
	}
	return fib(n)[0];

	function fib(n) {
		if (n === 0) {
			return [0n, 1n];
		} else {
			const [a, b] = fib(Math.floor(n / 2));
			const c = a * (b * 2n - a);
			const d = a * a + b * b;
			if (n % 2 === 0) {
				return [c, d];
			} else {
				return [d, c + d];
			}
		}
	}
}

console.log(fibonacci(100));
