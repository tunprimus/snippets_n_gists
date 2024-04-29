
let user = {}; // a user without "address" property

alert(user.address.street); // Error!

// document.querySelector('.elem') is null if there's no element
let html = document.querySelector('.elem').innerHTML; // error if it's null

/**
 * The optional chaining ?. stops the evaluation if the value before ?. is undefined or null and returns undefined.
 * In other words, value?.prop:
    works as value.prop, if value exists,
    otherwise (when value is undefined/null) it returns undefined.
 */
alert( user?.address?.street ); // undefined (no error)
html = document.querySelector('.elem')?.innerHTML; // will be undefined, if there's no element

alert(user.address ? user.address.street : undefined); // undefined
html = document.querySelector('.elem') ? document.querySelector('.elem').innerHTML : null;

/**
 * That’s just awful, one may even have problems understanding such code.
There’s a little better way to write it, using the && operator:
 */
alert(user.address ? user.address.street ? user.address.street.name : null : null); // null
alert( user.address && user.address.street && user.address.street.name ); // undefined (no error)

/**
 * Other variants: ?.(), ?.[]
 * The optional chaining ?. is not an operator, but a special syntax construct, that also works with functions and square brackets.
 * For example, ?.() is used to call a function that may not exist.
 */

let userAdmin = {
  admin() {
    alert("I am admin");
  }
};

let userGuest = {};

userAdmin.admin?.(); // I am admin

userGuest.admin?.(); // nothing happens (no such method)

/**
 * The ?.[] syntax also works, if we’d like to use brackets [] to access properties instead of dot .. Similar to previous cases, it allows to safely read a property from an object that may not exist.
 */

let key = "firstName";

let user1 = {
  firstName: "John"
};

let user2 = null;

alert( user1?.[key] ); // John
alert( user2?.[key] ); // undefined

/**
 * Also we can use ?. with delete:
 */
delete user?.name; // delete user.name if user exists

/**
 * We can use ?. for safe reading and deleting, but not writing
 * The optional chaining ?. has no use on the left side of an assignment.
 */
let user3 = null;

user3?.name = "John"; // Error, doesn't work
// because it evaluates to: undefined = "John"

/**
 * Summary
 * The optional chaining ?. syntax has three forms:

    obj?.prop – returns obj.prop if obj exists, otherwise undefined.
    obj?.[prop] – returns obj[prop] if obj exists, otherwise undefined.
    obj.method?.() – calls obj.method() if obj.method exists, otherwise returns undefined.
 * 
 */
