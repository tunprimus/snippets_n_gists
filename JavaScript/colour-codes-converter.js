//@ts-check

/**
 * Adapted from 30 Seconds of Code
 * @link https://www.30secondsofcode.org/js/s/rgb-hex-hsl-hsb-color-format-conversion
 */

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

/**
 * @description Convert RGB colour code to HSL with optional number array or string output
 * @example console.log(rgbToHsl(45, 23, 11)); // [ 21, 61, 11 ]
 * console.log(rgbToHsl(139, 0, 0)); // [ 0, 37, 27 ]
 * console.log(rgbToHsl(85, 107, 47)); // [ 82, 39, 30 ]
 * console.log(rgbToHsl(45, 23, 11, true)); // hsl(21, 61%, 11%)
 * console.log(rgbToHsl(139, 0, 0, true)); // hsl(0, 37%, 27%)
 * console.log(rgbToHsl(85, 107, 47, true)); // hsl(82, 39%, 30%)
 * @param {number} r 
 * @param {number} g 
 * @param {number} b 
 * @param {boolean} [strOutput=false] 
 * @returns {number[] | string}
 */
function rgbToHsl (r, g, b, strOutput = false) {
	r /= 255;
	g /= 255;
	b /= 255;

	const lig = Math.max(r, g, b);
	const sat = lig - Math.min(r, g, b);
	const hue = sat 
		? lig === r
			? (g - b) / sat
			: lig === g
			? 2 + (b - r) / sat
			: 4 + (r - g) / sat
		: 0;
	
	const tempArray = [
		60 * hue < 0 ? 60 * hue + 360 : 60 * hue, 
		100 * (sat ? (lig <= 0.5 ? sat / (2 * lig - sat) : sat / (2 - (2 * lig - sat))) : 0), 
		(100 * (2 * lig - sat)) / 2,
	];
	
	let result = tempArray.map(val => Math.round(val));
	
	return strOutput ? `hsl(${result[0]}, ${result[1] + '%'}, ${result[2] + '%'})` : result;
}

/**
 * @description Convert HSL to RGB with optional number array or string output
 * @example console.log(hslToRgb(13, 100, 11)); // [ 56, 12, 0 ]
 * console.log(hslToRgb(13, 100, 11, true)); // rgb(56, 12, 0)
 * @param {number} hue 
 * @param {number} sat 
 * @param {number} lig 
 * @param {boolean} [strOutput=false] 
 * @returns {number[] | string}
 */
function hslToRgb(hue, sat, lig, strOutput = false) {
	sat /= 100;
	lig /= 100;
	const k = n => (n + hue / 30) % 12;
	const a = sat * Math.min(lig, 1 - lig);
	const f = n => lig - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));

	let tempArray = [255 * f(0), 255 * f(8), 255 * f(4)];

	let result = tempArray.map(val => Math.round(val));

	return strOutput ? `rgb(${result[0]}, ${result[1]}, ${result[2]})` : result;
}

console.log(hslToRgb(13, 100, 11));
console.log(hslToRgb(13, 100, 11, true));
