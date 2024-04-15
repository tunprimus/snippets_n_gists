//@ts-check

/**
 * @description Get useful input values with formDataMap()
 * @author Dan Cătălin Burzo
 * @link https://danburzo.ro/formdatamap/
 * 
 * @param {HTMLFormElement} form 
 * @param {HTMLFormElement} submitter
 */
function formDataMap(form, submitter) {
	const excludedTags = ['FIELDSET', 'OBJECT', 'OUTPUT',];
	const excludedTypes = ['button', 'reset', 'image',];

	function shouldSubmit(el) {
		if (!el.name) {
			return false;
		}

		if (excludedTags.includes(el.tagName)) {
			return false;
		}

		if (excludedTypes.includes(el.type)) {
			return false;
		}

		if (el.type === 'submit' && el !== submitter) {
			return false;
		}

		if (el.type === 'checkbox' && !el.checked) {
			return false;
		}

		if (el.disabled || el.matches(':disabled')) {
			return false;
		}

		if (el.closest('datalist')) {
			return false;
		}
	}

	const result = {};

	function append(key, val) {
		result[key] = Object.prototype.hasOwnProperty.call(result, key) ?
			[].concat(result[key], val)
			: val;
	}

	Array.from(form.elements).forEach(el => {
		if (!shouldSubmit(el)) {
			return;
		}

		const { name, type } = el;
		if (type === 'number' || type === 'range') {
			append(name, +el.value);
		} else if (type === 'date' || type === 'datetime-local') {
			append(name, el.valueAsDate());
		} else if (type === 'file') {
			append(name, el.files);
		} else if (type === 'url') {
			append(name, new URL(el.value));
		} else if (type === 'select-one' || type === 'select-multiple') {
			Array.from(el.selectedOptions).forEach(option => append(name, option.value));
		} else {
			append(name, el.value);
		}
	});

	return result;
}

export { formDataMap };
