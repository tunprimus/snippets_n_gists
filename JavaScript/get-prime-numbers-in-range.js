//@ts-check
'use strict';

/**
 * @description Get prime numbers in a range
 * @example let testArr = getPrimeNumbersInRange(1000000);
 * console.log(testArr);
 * @param {number} upperLimit 
 * @returns {number[]}
 */
function getPrimeNumbersInRange(upperLimit) {
	// Generates a range of numbers
	let arr = [...Array(upperLimit + 1).keys()];
	// 1 is not a prime number
	arr[1] = 0;
	let squareRootLimit = Math.sqrt(upperLimit);
	for (let i = 2; i <= squareRootLimit; i++) {
		for (let j = i * i; j <= upperLimit; j += i) {
			arr[j] = 0;
		}
	}
	// Remove the zeroes
	return arr.filter(Number);
}

export { getPrimeNumbersInRange };
