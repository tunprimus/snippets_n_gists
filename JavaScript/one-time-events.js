//@ts-check
'use strict';


/**
 * One-time event generator
 *
 * @param {*} node
 * @param {string} type
 * @param {function} callback
 */
function oneTime(node, type, callback) {
	// Create event
	node.addEventListener(type, function (evt) {
		// Remove event listeners
		evt.target.removeEventListener(evt.type, arguments.callee);
		// Call handler
		return callback(evt);
	});
}

// One time event
oneTime(document.getElementById('myElement'), 'click', handler);

// Handler function

/**
 * Description
 * @param {event} evt
 * @returns {any}
 */
function handler(evt) {
	alert('You will only see this once!');
}
