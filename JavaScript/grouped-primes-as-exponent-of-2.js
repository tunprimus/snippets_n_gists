//@ts-check
'use strict';


/**
 * Prime number generator
 */
function* generatorPrimesObject() {
  let markedNotPrime = {};
  let valueToCheck = 2;

  while(true) {
    if (!(valueToCheck in markedNotPrime)) {
      yield valueToCheck;
      markedNotPrime[valueToCheck**2] = [valueToCheck];
    } else {
      let primes = markedNotPrime[valueToCheck];
      primes.forEach(prime => {
        let nextMultipleOfPrime = prime + valueToCheck;
        if (nextMultipleOfPrime in markedNotPrime) {
          markedNotPrime[nextMultipleOfPrime].push(prime);
        } else {
          markedNotPrime[nextMultipleOfPrime] = [prime];
        }
      });
      delete markedNotPrime[valueToCheck];
    }
    valueToCheck++;
  }
}

/**
 * Prime number 
 * @description Modified from https://benmccormick.org/2017/11/27/190000.html
 * @example console.log(generatePrime(1000));
 * console.log(generatePrime(1000, true));
 * @param {number} numOfPrimes 
 * @param {boolean} toJSON 
 * @returns 
 */
function generatePrime(numOfPrimes=97, toJSON=false) {
  let gen = generatorPrimesObject();
  let result = [];
  let resultJSON = {};
  let num = numOfPrimes;

  for (let i = 0; i < num; i++) {
    if (toJSON) {
      resultJSON[i] = gen.next().value;
    } else {
      result.push(gen.next().value);
    }
  }

  return toJSON ? JSON.stringify(resultJSON) : result;
}

let exponent = 1;
let topExponent = 3;
let buffer = {};

for (let i = exponent; i <= topExponent; i++) {
  buffer[`2^${i}`] = {value: 2 ** i, primes_in_segment: []};
}
console.log(buffer);

let counter = 1;
let primeBuffer;

function primeFetcher() {
  let lastEntry = 0;
  let upper2Exponent = 2 ** topExponent;
  let sqrtUpper2Exponent = Math.floor(Math.sqrt(upper2Exponent));
  while ((lastEntry <= upper2Exponent) || (counter < sqrtUpper2Exponent)) {
    primeBuffer = generatePrime(counter);
    counter++;
  }
  return primeBuffer;
}
primeFetcher()
