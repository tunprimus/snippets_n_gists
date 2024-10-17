//@ts-check
'use strict';


function wait(delay) {
	return new Promise(function(resolve) {
		setTimeout(resolve, delay);
	})
}

function fetchRetry(url, delay, tries, fetchOptions = {}) {
	function onError(err) {
		let triesLeft = tries - 1;
		if (!triesLeft) {
			throw err;
		}
		return wait(delay).then(function () {
			return fetchRetry(url, delay, triesLeft, fetchOptions);
		})
	}
	return fetch(url, fetchOptions).catch(onError);
}

export { fetchRetry, wait };
