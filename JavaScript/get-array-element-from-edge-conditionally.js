//@ts-check

/**
 * Get element from the start of an array while condition is met
 * @param {array} arr The array to extract from
 * @param {function} fn Callback function
 * @returns 
 */
function getFromStartWhile(arr, fn) {
	const index = arr.findIndex(n => !fn(n));
	return index === -1 ? arr : arr.slice(0, index);
}

/**
 * Get element from the end of an array while condition is met
 * @param {array} arr The array to extract from
 * @param {function} fn Callback function
 * @returns 
 */
function getFromEndWhile(arr, fn) {
	function findLastIndex (array, predicate) {
		let l = array.length;
		while (l--) {
			if (predicate(array[l], l, arr)) {
				return l;
			}
		}
		return -1;
	}
	let index = arr.findLastIndex(n => !fn(n))
	if (Number.isNaN(index)) {
		index = findLastIndex(arr, fn);
	}
	return index === -1 ? arr : arr.slice(index + 1);
}

export { getFromStartWhile, getFromEndWhile };
