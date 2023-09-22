/**
 * Inspired by
 * @title Testing a javascript 'Random Sampler'
 * @author wipdev
 * https://wipdev.hashnode.dev/testing-a-javascript-random-sampler
 */

const getRandomItem = (items) => items[Math.floor(Math.random() * items.length)];

// Test variables
const testArray = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',];
const counts = testArray.reduce((acc, cur) => {
  acc[cur] = 0;
  return acc;
}, {});
// console.log(counts);
// const counts2 = {a: 0, b: 0, c: 0, d: 0, e: 0, f: 0, g: 0, h: 0, i: 0, j: 0,};

let randomItem;
const SIMULATIONS = 99_000_000;

// Validation
for (let i = 0; i < SIMULATIONS; i++) {
  randomItem = getRandomItem(testArray);
  counts[randomItem]++;
}


// Result presentation
const proportions = {
  a: counts.a / SIMULATIONS,
  b: counts.b / SIMULATIONS,
  c: counts.c / SIMULATIONS,
  d: counts.d / SIMULATIONS,
  e: counts.e / SIMULATIONS,
  f: counts.f / SIMULATIONS,
  g: counts.g / SIMULATIONS,
  h: counts.h / SIMULATIONS,
  i: counts.i / SIMULATIONS,
  j: counts.j / SIMULATIONS,
};
// console.log(proportions);
console.table(proportions);


let proportions2 = [];
for (const prop in counts) {
  const collector = `${prop}: ${counts[prop] / SIMULATIONS}`;
  proportions2.push(collector);
}
console.table(proportions2);