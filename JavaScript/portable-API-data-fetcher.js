//@ts-check

let oldData = [
	{
		id: 1234,
		todo: 'Play D&D',
		completed: false,
	},
	{
		id: 5678,
		todo: 'Buy milk',
		completed: true,
	},
];

let updatedData = {
	1234: {
		todo: 'Play D&D',
		completed: false,
	},
	5678: {
		todo: 'Buy milk',
		completed: true,
	},
};

/**
 * Retrieves JSON data from the specified API endpoint and transforms it into a new format.
 *
 * @param {string} apiEndpoint - The URL of the API endpoint to fetch the data from.
 * @return {Promise<Array<Object>>} A Promise that resolves to an array of transformed objects.
 */
async function getAndTransformApiJsonData(apiEndpoint) {
	// Get the API data
	let request = await fetch(apiEndpoint);
	let returned = await request.json();

	// Transform the returned data
	let transformed = Object.entries(returned).map(([id, item]) => {
		let { neededData, status } = item;
		return {id, neededData, status};
	});

	// Return the transformed object
	return transformed;
}

/**
 * Retrieves API data from the specified URI and transforms it into a new format.
 *
 * @param {string} apiURI - The URI of the API to fetch the data from.
 * @return {Promise<Array<Object>>} A Promise that resolves to an array of transformed objects.
 * @example async function renderTodos() {
			let todos = await getApiData('/todos');
			let app = document.getElementById('app');
			// @ts-ignore
			app.innerHTML = `
				<ul>
					${todos.map((todo) => {
						return `<li>${todo}</li>`;
					}).join('')}
				</ul>
			`;
		}
 */
async function getApiData(apiURI) {
	let data = await getAndTransformApiJsonData(apiURI);
	return data;
}


export { getAndTransformApiJsonData, getApiData };
