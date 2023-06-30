/* Inspired from
https://github.com/sitepoint-editors/JOG-Maths
*/

export const jsFactorial = num => [...Array(num)].map((num, i) => i + 1).reduce((product, x) => product * x, 1);

export const jsRandomInt = (min, max) => max === undefined ? Math.floor(Math.random() * min) : min + Math.floor(Math.random() * (max - min + 1));

export const jsIsPrime = num => !(num < 2 || (num > 2 && num % 2 === 0) || [...Array(Math.floor(num**.5/2))].map((_, i) => i * 2 + 3).filter(x => num % x === 0).length);

export const jsFactors = num => [...Array(Math.abs(num))].map((_, i) => ++i).filter(x => num % x === 0);

export const jsEven = num => num % 2 === 0;

export const jsOdd = num => num % 2 !== 0;

export const jsSumTriangle = num => (num * (num + 1)) / 2;

export const jsSumOfSquares = num => (num * (num + 1) * (2 * num + 1)) / 6;

export const jsSumOfCubes = num => ((num ** 2) * ((num + 1) ** 2)) / 4;

export const jsDigitSum = num => num === 0 ? 0 : num % 9 || 9;

export const jsRoundUp = (num, decP = 0) => Number(num.toFixed(decP));

export const jsRoundSF = (num, sigF = 1) => Number(num.toPrecision(sigF));

export const jsGCD = (num1, num2) => num2 ? jsGCD(num2, num1 % num2) : num1;

export const jsHCF = (num1, num2) => num2 ? jsHCF(num2, num1 % num2) : num1;

export const jsLCM = (num1, num2) => (num1 * num2) / jsHCF(num1, num2);

const JSMathsPlus = {jsFactorial, jsRandomInt, jsIsPrime, jsFactors, jsEven, jsOdd, jsSumTriangle, jsSumOfSquares, jsSumOfCubes, jsDigitSum, jsRoundUp, jsRoundSF, jsGCD, jsHCF, jsLCM,};

export default JSMathsPlus;
