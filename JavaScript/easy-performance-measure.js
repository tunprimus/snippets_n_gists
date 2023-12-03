// @ts-check

/**
 * easy-performance-measure
 * @author mr-pascal
 * @link https://github.com/mr-pascal/easy-performance-measure
 */

/**
 * Measures the time in milliseconds it takes for the passed in method to finish
 * @param {function} fn The method which should be measured
 * @param {Array<unknown>} args A dynamic amount of parameters which will be passed into 'fn'
 * @return {Promise<[unknown, number]>} A tuple where the first element is the response of the passed in method and the
 * second the time in milliseconds it took to complete
 */
export const measureAsync = async (fn, ...args) => {
  const startTime = new Date().getTime();

  const fnResult = await fn(...args);

  const endTime = new Date().getTime();

  const durationInMs = endTime - startTime;

  return [fnResult, durationInMs];
};

/**
 * Measures the time in milliseconds it takes for the passed in method to finish
 * @param {function} fn The method which should be measured
 * @param {Array<unknown>} args A dynamic amount of parameters which will be passed into 'fn'
 * @return {[unknown, number]} A tuple where the first element is the response of the passed in method and the
 * second the time in milliseconds it took to complete
 */
export const measureSync = (fn, ...args) => {
  const startTime = new Date().getTime();

  const fnResult = fn(...args);

  const endTime = new Date().getTime();

  const durationInMs = endTime - startTime;

  return [fnResult, durationInMs];
};
