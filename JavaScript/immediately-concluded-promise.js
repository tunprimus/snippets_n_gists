//@ts-check


function getUsers(userIds) {
	if (userIds === null || !Array.isArray(userIds)) {
		return Promise.reject(new Error('User IDs must be an array'));
	}

	if (userIds.length === 0) {
		return Promise.resolve([]);
	}

	return Promise.allSettled(userIds.map(id => getUser(id)))
		.then(results => {
			return results
				.filter(result => result.status === 'fulfilled')
				.map(result => result.value);
		})
}