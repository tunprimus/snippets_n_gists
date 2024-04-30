//@ts-check

/**
 * Deep clone an object using a custom function
 * @param {object} input 
 * @returns {object}
 */
function deepClone(input) {
	if (input === null || typeof input !== 'object') {
		return input;
	}

	const initialValue = Array.isArray(input) ? [] : {};
	return Object.keys(input).reduce((acc, key) => {
		acc[key] = deepClone(input[key]);
		return acc;
	}, initialValue);
}

export { deepClone };
