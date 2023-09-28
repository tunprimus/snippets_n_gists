/**
 * @author Chris Ferdinandi
 * @description Getting the true type of an object by using the call function
 * @params Pass in any object
 * @returns {String} Returns string from Object.prototype
 */

var trueTypeOf = function (obj) {
  return Object.prototype.toString.call(obj).slice(8, -1).toLowerCase();
};

export { trueTypeOf };
