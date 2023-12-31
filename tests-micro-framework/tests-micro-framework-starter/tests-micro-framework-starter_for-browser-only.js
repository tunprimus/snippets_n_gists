/*
  Inspired by:
  Testing JavaScript without a (third-party) framework +++>>> https://alexwlchan.net/2023/testing-javascript-without-a-framework/
  Unit Test Your JavaScript Code Without a Framework +++>>> https://javascript.plainenglish.io/unit-test-front-end-javascript-code-without-a-framework-8f00c63eb7d4
*/
  
/**
 * test function
 * @param {string} description
 * @param {function} test_function
 */

function it(description, test_function) {
  const attachResult = document.querySelector('#attach-result');
  const result = document.createElement('p');
  result.classList.add('test-result');

  try {
    test_function();
    result.classList.add('success');
    result.textContent = description;
  } catch (err) {
    result.classList.add('failure');
    result.innerHTML = `${description}<br><pre>${err}</pre>`;
  }

  attachResult.appendChild(result);
}

function assertEqual(expected, actual) {
  if (expected === actual || (typeof expected === 'object' && typeof actual === 'object' && expected.length === actual.length && expected.every((element, index) => element === actual[index]))) {
    return;
  } else {
    throw new Error(`${expected} != ${actual}`);
  }
}

function assertTrue(value) {
  assertEqual(value, true);
}

function assertFalse(value) {
  assertEqual(value, false);
}

function assert(isTrue) {
  if (!isTrue) {
    throw new Error();
  }
}

/* Demo */
/* 
it('should fail', function() {
  assert(1 !== 1);
});
 */
