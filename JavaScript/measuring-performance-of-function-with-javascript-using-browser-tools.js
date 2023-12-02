// @ts-check

/**
 * Measuring the performance of a function with Javascript using browser tools or creating your own benchmark
 * @author Carlos Delgado
 * @link https://ourcodeworld.com/articles/read/144/measuring-the-performance-of-a-function-with-javascript-using-browser-tools-or-creating-your-own-benchmark
 */

/**
 * Measure the time of execution with Date timestamp of a synchronous task
 * Old benchmark
 * @param {Function} toMeasure
 * @param {number} repeatTimes
 * @returns {object}
 */
function TimeBenchmark(toMeasure, repeatTimes) {
  if (typeof(repeatTimes) !== 'number') {
    repeatTimes = 1;
  }

  if (typeof(toMeasure) === 'function') {
    var start_time = new Date().getTime();
    for (var i = 0; i < repeatTimes; i++) {
      toMeasure.call();
    }
    var end_time = new Date().getTime();
  }

  return {
    start: start_time,
    end: end_time,
    estimatedMilliseconds: end_time - start_time,
  };
}

/*
// Using Old Benchmark
var TimesToBeExecuted = 10;
var TaskToBeExecuted = function() {
  // A task that you want to measure
};
var TestResult = new TimeBenchmark(TaskToBeExecuted, TimesToBeExecuted);
console.log(TestResult);
*/

/**
 * 
 * @param {Function} toMeasure 
 * @param {number} repeatTimes 
 * @returns {object}
 */
function StandardBenchmark(toMeasure, repeatTimes) {
  if (typeof(repeatTimes) !== 'number') {
    repeatTimes = 1;
  }

  if (typeof(toMeasure) === 'function') {
    var start_status = performance.now();
    var total_taken = 0;
    for (var i = 0; i < repeatTimes; i++) {
      var startTimeSubtask = performance.now();
      toMeasure.call();
      var endTimeSubtask = performance.now();
      total_taken += (endTimeSubtask - startTimeSubtask);
    }
    var final_status = performance.now();
  }

  return {
    totalMilliseconds: (final_status - start_status),
    averageMillisecondsPerTask: total_taken / repeatTimes,
  };
}

/*
// Using Standard Benchmark
var TimesToBeExecuted = 10;
var TaskToBeExecuted = function() {
  // A task that you want to measure
};
var TestResult = new StandardBenchmark(TaskToBeExecuted, TimesToBeExecuted);
console.log(TestResult);
*/

/**
 * 
 * @param {Function} toMeasure 
 * @param {string} identifierName 
 * @param {number} repeatTimes 
 * @returns {object}
 */
function ConsoleTimeBenchmark(toMeasure, identifierName, repeatTimes) {
  if (!identifierName) {
    identifierName = new Date().getTime().toString();
  }

  if (typeof(repeatTimes) !== 'number') {
    repeatTimes = 1;
  }

  if (typeof(toMeasure) === 'function') {
    console.time(identifierName);
    for (var i = 0; i < repeatTimes; i++) {
      toMeasure.call();
    }
    console.timeEnd(identifierName);
    return true;
  }

  return false;
}

/**
 * 
 * @param {Function} toMeasure 
 * @param {string} identifierName 
 * @param {number} repeatTimes 
 * @returns {Promise}
 */
async function UserTimingBenchmark(toMeasure, identifierName, repeatTimes) {
  if (!identifierName) {
    identifierName = new Date().getTime().toString();
  }

  if (typeof(repeatTimes) !== 'number') {
    repeatTimes = 1;
  }

  let startMarker = `Start-${identifierName}`;
  let endMarker = `End-${identifierName}`;
  let measurementName = `Measuring-${identifierName}`;

  if (typeof(toMeasure) === 'function') {
    performance.mark(startMarker);
    for (var i = 0; i < repeatTimes; i++) {
      await toMeasure.call();
    }
    performance.mark(endMarker);
  }

  performance.measure(measurementName, startMarker, endMarker);

  var marks = performance.getEntriesByType('mark');
  var measurements = performance.getEntriesByType('measure');
  console.log(marks, measurements);
  return measurements;
}
