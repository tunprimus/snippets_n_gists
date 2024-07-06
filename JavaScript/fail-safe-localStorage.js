//@ts-check

/**
 * Retrieves the value associated with the specified key from the local storage.
 *
 * @param {string} key - The key to retrieve the value for.
 * @return {any | undefined} - The value associated with the key, or undefined if the key does not exist or if there was an error parsing the value.
 */
export function getItem(key) {
  try {
    let _JSON$parse;
		let _window$localStorage$;
    return (_JSON$parse = JSON.parse(
      (_window$localStorage$ = window.localStorage.getItem(key)) !== null &&
        _window$localStorage$ !== void 0 ? _window$localStorage$ : "null"
    )) !== null && _JSON$parse !== void 0 ? _JSON$parse : undefined;
  } catch {
    return undefined;
  }
};

/**
 * Sets the value for the specified key in the local storage.
 *
 * @param {string} key - The key to set the value for.
 * @param {any} value - The value to be stored.
 * @return {void} 
 */
export function setItem(key, value) {
  try {
    return window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* NOOP */
  }
};

/**
 * Removes the item with the specified key from the local storage. If the key does not exist or if there is an error,
 * the function does nothing.
 *
 * @param {string} key - The key of the item to be removed.
 * @return {void} 
 */
export function removeItem(key) {
  try {
    window.localStorage.removeItem(key);
  } catch {
    /* NOOP */
  }
};

