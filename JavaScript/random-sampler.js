/**
 * Inspired by
 * @title Testing a javascript 'Random Sampler'
 * @author wipdev
 * https://wipdev.hashnode.dev/testing-a-javascript-random-sampler
 */

const getRandomItem = (items) => items[Math.floor(Math.random() * items.length)];

// Test variables
const testArray = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',];
const counts = {a: 0, b: 0, c: 0, d: 0, e: 0, f: 0, g: 0, h: 0, i: 0, j: 0,};
let randomItem;
const simulations = 99_000_000;

// Validation
for (let i = 0; i < simulations; i++) {
  randomItem = getRandomItem(testArray);
  counts[randomItem]++;
}

// Result presentation
const proportions = {
  a: counts.a / simulations,
  b: counts.b / simulations,
  c: counts.c / simulations,
  d: counts.d / simulations,
  e: counts.e / simulations,
  f: counts.f / simulations,
  g: counts.g / simulations,
  h: counts.h / simulations,
  i: counts.i / simulations,
  j: counts.j / simulations,
};
console.log(proportions);
console.table(proportions);
