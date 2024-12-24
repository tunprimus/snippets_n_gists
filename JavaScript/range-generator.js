//@ts-check
'use strict';

/**
 * Create a range of numbers.
 * https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/from#sequence_generator_range
 * @param {number} stop The last number in the range.
 * @param {number} start The first number in the range.
 * @param {number} step The step between each number in the range.
 * @returns {number[]} A range of numbers.
 * @example console.log(generateRange(10)); // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
 */
export function generateRange(stop, start = 1, step = 1) {
	return Array.from({ length: (stop - start) / step + 1 }, (_, i) => start + i * step);
}
