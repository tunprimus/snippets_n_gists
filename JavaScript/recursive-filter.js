//@ts-check
/**
 * @description Filter a nested object recursively
 * @author  Dave Cohen
 * @link https://www.scraggo.com/recursive-filter-function/
 */

function log(item) {
	console.log(JSON.stringify(item));
}

const people = [
	{
		age: 2,
	},
	{
		age: 22,
		friends: [
			{
				age: 17,
			},
			{
				age: 34,
			},
			{
				name: "Sherry O'Teri",
			}
		],
	},
	{
		name: 'Bob Ross',
	},
	{
		age: 12,
		friends: [
			{
				age: 44,
			},
		],
	}
];

function validAge(item) {
	return typeof item.age === 'number' && item.age > 17;
}

function thisWontWorkRecursiveFilter(arr) {
	return arr.filter(item => {
		if (item.friends) {
			// Here is the recursive call
			return (thisWontWorkRecursiveFilter(item.friends).length > 0 && validAge(item));
		} else {
			return validAge(item);
		}
	});
}
log(thisWontWorkRecursiveFilter(people));

function recursiveFilter(arr) {
	return arr.reduce((acc, item) => {
		const newItem = item;

		if (item.friends) {
			// Here is the recursive call
			newItem.friends = recursiveFilter(item.friends);
		}
		if (validAge(newItem)) {
			// Here acc takes the new item
			acc.push(newItem);
		}
		// Always return acc
		return acc;
	}, []);
}
log(recursiveFilter(people));
