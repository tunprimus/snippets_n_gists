//@ts-check


class HeaderPromise {
	constructor(uri) {
		this.uri = uri;
		this.headers = {}
	}

	withHeader(key, value) {
		this.headers[key] = value;
		return this;
	}

	then(onResolved, onRejected) {
		return this.getInnerPromise().then(onResolved, onRejected);
	}

	getInnerPromise() {
		if (!this.innerPromise) {
			this.innerPromise = (async() => {
				const response = await fetch(this.uri, {
					headers: this.headers
				});
				const json = await response.json() || await response.text();
				return json;
			})();
		}
		return this.innerPromise;
	}
}

const ARTICLE_URI = 'https://api.example';
function getArticles() {
	return new HeaderPromise(ARTICLE_URI);
}
