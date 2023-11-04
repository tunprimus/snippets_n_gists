/**
 * @function isPrime
 * @description Checks for prime number with a fast algorithm by skipping lots of factors and multiples
 * @param {Number} num - Number to check
 * @return {Boolean} true if prime
 */

function isPrime(num) {
  if (typeof num !== 'number' || Number.isNaN(num)) {
    throw new TypeError('Parameter is not a number!');
  }

  if (num <= 1) {
    return false;
  }

  if (num <= 3) {
    return true;
  }

  if (num % 2 === 0 || num % 3 === 0) {
    return false;
  }

  for (var i = 5; i * i <= num; i += 6) {
    if (num % i === 0 || num % (i + 2) === 0) {
      return false;
    }
  }
  return true;
}

export { isPrime };
