//@ts-check

/**
 * @description Convert RGB colour code to hex string using the << bitwise left-shift operator
 * @example console.log(rgbToHex(255, 165, 1)); // #FFA501
 * @param {number} r 
 * @param {number} g 
 * @param {number} b 
 * @returns {string}
 */
function rgbToHex(r, g, b) {
	const temp = ((r << 16) + (g << 8) + b).toString(16).padStart(6, '0');
	return '#' + (temp.toUpperCase());
}


/**
 * @description Convert hexadecimal to RGB colour code using the >> bitwise right-shift operator
 * @example console.log(hexToRgb('#27AE60FF')); // rgba(39, 174, 96, 255)
 * console.log(hexToRgb('27AE60')); // rgb(39, 174, 96)
 * console.log(hexToRgb('#FFF')); // rgb(255, 255, 255)
 * @param {string} hex 
 * @returns 
 */
function hexToRgb(hex) {
	let alpha = false;
	let hexString = hex.slice(hex.startsWith('#') ? 1 : 0);

	if (hexString.length === 3) {
		hexString = [...hexString].map(x => x + x).join('');
	} else if (hexString.length === 8) {
		alpha = true;
	}

	let h = parseInt(hexString, 16);

	return (
		'rgb' + 
		(alpha ? 'a' : '') + 
		'(' + 
		(h >>> (alpha ? 24 : 16)) + 
		', ' + 
		((h & (alpha ? 0x00FF0000 : 0x00FF00)) >>> (alpha ? 16 : 8)) + 
		', ' + 
		((h & (alpha ? 0x0000FF00 : 0x0000FF)) >>> (alpha ? 8 : 0)) + 
		(alpha ? `, ${h & 0x000000FF}` : '') + 
		')'
	);
}


