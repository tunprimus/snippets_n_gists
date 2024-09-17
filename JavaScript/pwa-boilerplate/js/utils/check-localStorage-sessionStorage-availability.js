//@ts-check

/**
 * 
 * @returns {boolean}
 */
function isLocalStorageEnabled() {
	try {
		const key = `__storage__test`;
		window.localStorage.setItem(key, null);
		window.localStorage.removeItem(key);
		return true;
	} catch (err) {
		return false;
	}
}

console.log(isLocalStorageEnabled());

/**
 * 
 * @returns {boolean}
 */
function isSessionStorageEnabled() {
	try {
		const key = `__storage__test`;
		window.sessionStorage.setItem(key, null);
		window.sessionStorage.removeItem(key);
		return true;
	} catch (err) {
		return false;
	}
}

console.log(isSessionStorageEnabled());
