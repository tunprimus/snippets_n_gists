/**
 * Inspired by Why You Should Be Using globalThis Instead of Window In Your Javascript Code #+++>>> https://ilikekillnerds.com/2023/02/why-you-should-be-using-globalthis-instead-of-window-in-your-javascript-code/
 */


// In browser
console.log(globalThis === window); // true
console.log(globalThis.location.href); // same as window.location.href


// In NodeJS
console.log(globalThis === global); // true
console.log(globalThis.setTimeout === global.setTimeout); // true


// In web worker
console.log(globalThis === self); // true
console.log(globalThis.postMessage === self.postMessage); // true


// In unit tests
function showMessage(message) {
  globalThis.alert(message);
}

it('shows message', () => {
  const originalAlert = globalThis.alert;
  globalThis.alert = jest.fn();
  showMessage('Hello, world!');
  expect(globalThis.alert).toHaveBeenCalledWith('Hello, world!');
  globalThis.alert = originalAlert;
});
