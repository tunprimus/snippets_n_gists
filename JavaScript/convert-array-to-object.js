//@ts-check

/**
 * Convert array to object using the indices as keys and array elements as values
 * @param {Array} arr 
 * @returns {object}
 */
function convertArrayToObject(arr) {
	return arr.reduce((acc, currentVal, i) => ({
		...acc, [i]: currentVal}), {});
}

export { convertArrayToObject };
