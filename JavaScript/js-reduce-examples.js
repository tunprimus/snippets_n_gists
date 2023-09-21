/*  */

/* Reduce method signature
  array.reduce(callback(accumulator, currentValue, index, array), initialValue);
  callback - called for each element in the array
  accumulator - accumulated value so far
  currentValue - value of the current element in the array
  index - index of the current element in the array (optional)
  array - array on which reduce is called (optional)
  initialValue - optional value for starting the accumulator; if not provided, the first element of the array will be used
*/

// Basic usage
const numbers = [1, 2, 3, 4, 5];

const sum = numbers.reduce((acc, current) => acc + current, 0);
console.log(sum);

// Grouping elements of an array
const students = [
  {name: 'Alice', class: 'A',},
  {name: 'Bob', class: 'B',},
  {name: 'Charlie', class: 'A',},
];

const groupByClass = students.reduce((acc, current) => {
  const className = current.class;
  if (!acc[className]) {
    acc[className] = [];
  }

  acc[className].push(current);
  return acc;
}, {});
console.log(groupByClass);

// Calculating statistics
const data = [10, 20, 30, 40, 50,];

const stats = data.reduce((acc, current) => {
  acc.sum += current;
  acc.squareSum += current * current;
  return acc;
}, { sum: 0, squareSum: 0 });

const mean = stats.sum / data.length;
const variance = stats.squareSum / data.length - mean * mean;
const stdDeviation = Math.sqrt(variance);
console.log(mean, stdDeviation);
console.log(mean);
console.log(stdDeviation);

// Building an object from an array
const pairs = [
  ['a', 1],
  ['b', 2],
  ['c', 3],
];

const obj = pairs.reduce((acc, [key, value]) => {
  acc[key] = value;
  return acc;
}, {});
console.log(obj);

// Find the longest sequence of identical characters
const text = 'aaabbccccccdddddddd';

const longestSequence = text.split('').reduce((acc, current) => {
  if (current === acc.currentChar) {
    acc.currentCount++;
    if (acc.currentCount > acc.maxCount) {
      acc.maxCount = acc.currentCount;
      acc.maxLength = acc.currentCount * current.length;
    }
  } else {
    acc.currentChar = current;
    acc.currentCount = 1;
  }
  return acc;
}, { currentChar: '', currentCount: 0, maxCount: 0, maxLength: 0 });
console.log(longestSequence);

// Managing complex database queries
const filters = [
  {field: 'age', operator: '>', value: 30},
  {field: 'country', operator: '=', value: 'USA'},
];
const query = filters.reduce((acc, filter, index) => {
  const condition = `${filter.field} ${filter.operator} "${filter.value}"`;
  console.log(condition);
  if (index === 0) {
    return `SELECT * FROM table WHERE ${condition}`;
  } else {
    return `${acc} AND ${condition}`;
  }
}, '');
console.log(query);

// Building a pipe
const users = [
  {id: 1, name: 'Alice', age: 28, country: 'USA',},
  {id: 2, name: 'Bob', age: 35, country: 'Canada',},
  {id: 3, name: 'Charlie', age: 22, country: 'USA',},
  {id: 4, name: 'David', age: 40, country: 'UK',},
  {id: 5, name: 'Eve', age: 30, country: 'Canada',},
];

const filters2 = [
  (users) => users.filter((user) => user.age >= 30),
  (users) => users.filter((user) => user.country === 'USA'),
];

const filteredUsers = filters2.reduce((acc, filter) => filter(acc), users);
console.log(filteredUsers);
