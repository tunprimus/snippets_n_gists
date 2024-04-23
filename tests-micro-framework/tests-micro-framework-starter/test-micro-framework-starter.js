/*
  Inspired by:
  Testing JavaScript without a (third-party) framework +++>>> https://alexwlchan.net/2023/testing-javascript-without-a-framework/
  Unit Test Your JavaScript Code Without a Framework +++>>> https://javascript.plainenglish.io/unit-test-front-end-javascript-code-without-a-framework-8f00c63eb7d4
*/
  
/**
 * test function
 * @param {string} description
 * @param {function} functionToTest
 */

function it(description, functionToTest) {
  const attachResult = document.querySelector('#attach-result');
  const result = document.createElement('p');
  result.classList.add('test-result');

  try {
    functionToTest();
    result.classList.add('success');
    result.innerHTML = description;
    console.log('\x1b[32m%s\x1b[0m', '\u2714 ' + description);
  } catch (err) {
    result.classList.add('failure');
    result.innerHTML = `${description}<br><pre>${err}</pre>`;
    console.log('\n');
    console.log('\x1b[31m%s\x1b[0m', '\u2718 ' + description);
    console.error(err);
  }

  attachResult.appendChild(result);
  // document.body.appendChild(result);
}

/**
 * Callback functions to use in the `it` function above
 */
function assertEqual(output, expected) {
  if (output === expected || (typeof output === 'object' && typeof expected === 'object' && output.length === expected.length && output.every((element, index) => element === expected[index]))) {
    return;
  } else {
    throw new Error(`${output} != ${expected}`);
  }
}

function assertTrue(value) {
  assertEqual(value, true);
}

function assertFalse(value) {
  assertEqual(value, false);
}

/**
 * 
 * @param {condition} isTrue 
 */
function assert(isTrue) {
  if (!isTrue) {
    throw new Error();
  }
}

/**
 * Demo
 * @example
 * it('should fail', function() {
 *  assert(1 !== 1);
 * });
 * 
 * it('should pass', function() {
 *  assert(1 === 1);
 * });
 */

it('should fail', function() {
  assert(1 !== 1);
});

it('should pass', function() {
  assert(1 === 1);
});

it('should fail', function() {
  assertTrue(1 !== 1);
});

it('should pass', function() {
  assertTrue(1 === 1);
});

it('should fail', function() {
  assertFalse(1 === 1);
});

it('should pass', function() {
  assertFalse(1 !== 1);
});
