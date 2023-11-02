/**
 * @author 30 Seconds of Code
 */

/**
 * 
 * @param {Function} fn - Function to pass in
 * @param {Number} ms - Number of milliseconds to delay
 */
const debounce = (fn, ms = 0) => {
  let timeoutId;
  return function (/** @type {any} */ ...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), ms);
  };
};

window.addEventListener('resize', debounce(() => {
  console.log(window.innerWidth);
  console.log(window.innerHeight);
}, 250));
