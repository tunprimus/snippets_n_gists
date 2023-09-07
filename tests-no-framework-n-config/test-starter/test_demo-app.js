/*
  Inspired by:
  Testing JavaScript without a (third-party) framework +++>>> https://alexwlchan.net/2023/testing-javascript-without-a-framework/
  Unit Test Your JavaScript Code Without a Framework +++>>> https://javascript.plainenglish.io/unit-test-front-end-javascript-code-without-a-framework-8f00c63eb7d4
*/

'use strict';
  
/**
 * test function
 * @param {string} description
 * @param {function} test_function
 */

function it(description, test_function) {
  const result = document.createElement('p');
  result.classList.add('test-result');

  try {
    test_function();
    result.classList.add('success');
    result.textContent = description;
    console.log('\x1b[32m%s\x1b[0m', '\u2714 ' + description);
  } catch (err) {
    result.classList.add('failure');
    result.innerHTML = `${description}<br><pre>${err}</pre>`;
    console.log('\n');
    console.log('\x1b[31m%s\x1b[0m', '\u2718 ' + description);
    console.error(err);
  }

  document.body.appendChild(result);
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

/* Test Demo App */
it('should validate a date string', function () {
  // Valid Date
  assert($myApp.isValidDate('02/02/2020'));
  // Invalid Date
  assert(!$myApp.isValidDate('01/32/2020'));
});

// Test Case for DOM
it('should add a todo item to the list', function() {
  var selector = document.querySelector('#selector');
  selector.innerHTML = '<form id="aform"><input type="text" name="todo-input"><button>Submit</button></form><ul id="todo-list"></ul>';
  var form = document.querySelector('#aform');
  form.elements['todo-input'].value = 'task 1';

  var evt = document.createEvent('HTMLEvents');
  evt.initEvent('submit', true, true);
  form.dispatchEvent(evt);

  assert(selector.innerHTML.toLowerCase().includes('<li>task 1</li>'));

  // Cleanup
  selector.innerHTML = '';
});

// Stub the Asynchronous Code
it('should get render user details object', function() {
  // Stub the get function
  $myApp.get = function(url, callback) {
    if (url === '/api/users?id=1') {
      callback({ fname: 'Amit', lname: 'Gupta', });
    } else {
      callback(false);
    }
  };

  // Attach #user-detail div to the test dom
  var selector = document.querySelector('#selector');
  selector.innerHTML = '<div id="user-detail"></div>';

  $myApp.getUser(1);
  assert(selector.innerHTML.includes('Amit Gupta'));
  selector.innerHTML = '';
});
