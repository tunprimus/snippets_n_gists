

/**
 * The result of a ?? b is:
    if a is defined, then a,
    if a isn’t defined, then b.
 * In other words, ?? returns the first argument if it’s not null/undefined. Otherwise, the second one.
 * The common use case for ?? is to provide a default value.
 */
let resultNullish = a ?? b;

let resultNonNullish = (a !== null && a !== undefined) ? a : b;

/**
 * We can also use a sequence of ?? to select the first value from a list that isn’t null/undefined.
 */

let firstName = null;
let lastName = null;
let nickName = "Supercoder";

// shows the first defined value:
alert(firstName ?? lastName ?? nickName ?? "Anonymous"); // Supercoder

/**
 * The OR || operator can be used in the same way as ??, as it was described above.
 */
// shows the first truthy value:
alert(firstName || lastName || nickName || "Anonymous"); // Supercoder

/**
 * The important difference between them is that:
    || returns the first truthy value.
    ?? returns the first defined value.
 * In other words, || doesn’t distinguish between false, 0, an empty string "" and null/undefined. They are all the same – falsy values. If any of these is the first argument of ||, then we’ll get the second argument as the result.
 * In practice though, we may want to use default value only when the variable is null/undefined. That is, when the value is really unknown/not set.
 */

let height = 0;

alert(height || 100); // 100
alert(height ?? 100); // 0

/**
 * The height || 100 checks height for being a falsy value, and it’s 0, falsy indeed.
 *	- so the result of || is the second argument, 100.
 * The height ?? 100 checks height for being null/undefined, and it’s not,
 * 	- so the result is height “as is”, that is 0.
 */
