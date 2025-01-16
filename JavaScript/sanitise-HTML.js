//@ts-check
'use strict';


/**
 * Sanitise HTML input by stripping out characters with special meanings
 *
 * @param {string} str
 * @returns {string}
 */
function sanitiseHTML(str) {
	return str.replace(/[^w, ]/gi, function (char) {
		return `&#${char.charCodeAt(0)}`;
	});
}

export {sanitiseHTML};
