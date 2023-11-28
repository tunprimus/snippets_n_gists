// @ts-check

/**
 * 
 * @param {Node} elem - The node to add event listeners to
 * @param {string} event - The event to add to the node
 * @param {Function} fn - The function to call when the event is emitted
 * @param {object} opts - Additional options
 * @returns {Function}
 */
const eventAdder = (elem, event, fn, opts = {}) => {
  const delegatorFn = evt => evt.target.matches(opts.target) && fn.call(evt.target, evt);
  elem.addEventListener(
    event,
    // @ts-ignore
    opts.target ? delegatorFn : fn,
    opts.options || false
  );
  if (opts.target) {
    return delegatorFn;
  }
};

/**
 * 
 * @param {Node} elem - The node to remove event listeners to
 * @param {string} event - The event to add to the node
 * @param {Function} fn - The function to call when the event is emitted
 * @param {object} opts - Additional options
 */
// @ts-ignore
const eventRemover = (elem, event, fn, opts = false) => elem.removeEventListener(event, fn, opts);