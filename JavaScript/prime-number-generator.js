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
 * Prime number generator
 */
function* generatorPrimesMap() {
  let markedNotPrimeMap = new Map();
  let valueToCheck = 2;

  while(true) {
    if (!(valueToCheck in markedNotPrimeMap)) {
      yield valueToCheck;
      markedNotPrimeMap.set(valueToCheck**2, [valueToCheck]);
    } else {
      let primes = markedNotPrimeMap.get(valueToCheck);
      primes.forEach(prime => {
        let nextMultipleOfPrime = prime + valueToCheck;
        if (markedNotPrimeMap.has(nextMultipleOfPrime)) {
          markedNotPrimeMap.get(nextMultipleOfPrime).push(prime);
        } else {
          markedNotPrimeMap.set(nextMultipleOfPrime, [prime]);
        }
      });
      markedNotPrimeMap.delete(valueToCheck);
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
function generatePrime(numOfPrimes=1000000, toJSON=false) {
  let gen = generatorPrimesObject() || generatorPrimesMap();
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

export { generatePrime };
