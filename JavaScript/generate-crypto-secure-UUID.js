//@ts-check

/**
 * @description Generating Cryptographically Secure Random Numbers in Vanilla JavaScript
 * @link https://maxpelic.com/blog/post/crypto-js/
 * @author Maxwell Pelic
 */
const PASSWORD_CHARS = `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+ -=[]{}|;':,./<>?`;

/**
 * @description Generates a random number in a given range
 * @param {number} minimum 
 * @param {number} maximum 
 * @returns {number}
 */
const randomRange = (minimum, maximum) => {
	if (!window.crypto) {
		throw new Error('Crypto library not available!');
	}

	/**
	 * Check that the provided values are valid
	 */
	if (minimum >= maximum) {
		throw new Error('Minimum must be less than maximum!');
	}

	if (maximum - minimum > 255) {
		throw new Error('Maximum range must be less than 256!');
	}

	/**
	 * Get random byte
	 */
	let randomByte = new Uint8Array(1);
	crypto.getRandomValues(randomByte);

	const result = randomByte[0] + minimum;

	/**
	 * Rejection sampling
	 */
	if (result > maximum) {
		return randomRange(minimum, maximum);
	} else {
		return result;
	}
};


const randomString = (length) => {
	/**
	 * Check that the provided values are valid
	 */
	if (length < 1) {
		throw new Error('Length must be greater than 0!');
	}

	let result = '';
	for (let i = 0; i < length; i++) {
		result += PASSWORD_CHARS[randomRange(0, PASSWORD_CHARS.length - 1)];
	}
	return result;
};

const password = randomString(16);
console.log(password);
