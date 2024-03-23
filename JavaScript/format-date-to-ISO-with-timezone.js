//@ts-check

const pad = n => `${Math.floor(Math.abs(n))}`.padStart(2, '0');

/**
 * 
 * @param {Date} date 
 * @returns {string}
 */
const getTimezoneOffset = date => {
	const tzOffset = -date.getTimezoneOffset();
	const diff = tzOffset >= 0 ? '+' : '-';
	return diff + pad(tzOffset / 60) + ':' + pad(tzOffset % 60);
};

/**
 * 
 * @param {Date} date 
 * @returns {string}
 */
const toISOStringWithTimezone = date => {
	return date.getFullYear() +
	'-' + pad(date.getMonth() + 1) +
	'-' + pad(date.getDate()) +
	'T' + pad(date.getHours()) +
	':' + pad(date.getMinutes()) +
	':' + pad(date.getSeconds()) +
	getTimezoneOffset(date);
};

/**
 * Validate ISO string
 * @param {string} val String to check for
 */
const isISOString = val => {
	const d = new Date(val);
	return !Number.isNaN(d.valueOf()) && d.toISOString() === val;
};


const isISOStringWithTimezone = val => {
	const d = new Date(val);
	return !Number.isNaN(d.valueOf()) && toISOStringWithTimezone(d) === val;
};

export { getTimezoneOffset, toISOStringWithTimezone, isISOString, isISOStringWithTimezone };
