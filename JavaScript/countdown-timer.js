//@ts-check

/**
 * Initializes a countdown timer on the specified element and updates it every second until the specified allotted time has elapsed.
 *
 * @param {string} nodeName - The CSS selector of the element to display the countdown timer on.
 * @param {number} allottedTime - The time in milliseconds for the countdown timer to run.
 * @return {void} This function does not return anything.
 */
export function countdown(nodeName, allottedTime) {
	let element;
	let endTime;
	let hours;
	let minutes;
	let seconds;
	let timeLeft;

	/**
 * Converts a number to a two-digit string by adding a leading zero if necessary.
 *
 * @param {number} n - The number to be converted to a two-digit string.
 * @return {string} The two-digit string representation of the input number.
 */
	function makeTwoDigits(n) {
		return (n < 10 ? '0' : '') + n;
	}

		/**
	 * Updates the timer on the element with the remaining time until the end time.
	 *
	 * @return {void} This function does not return anything.
	 */
	function updateTimer() {
		timeLeft = endTime - Date.now();

		if (timeLeft <= 0) {
			clearInterval(element.timer);
			return;
		}

		hours = Math.floor(timeLeft / 3600000);
		minutes = Math.floor((timeLeft - hours * 3600000) / 60000);
		seconds = Math.floor((timeLeft - hours * 3600000 - minutes * 60000) / 1000);
		element.textContent = `${makeTwoDigits(hours)}:${makeTwoDigits(minutes)}:${makeTwoDigits(seconds)}`;
	}

	element = document.querySelector(nodeName);
	endTime = Date.now() + allottedTime;
	
	updateTimer();
	element.timer = setInterval(updateTimer, 1000);
}