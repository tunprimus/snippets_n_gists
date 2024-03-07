
const routes = [
	{path: '/', callback: () => console.log('Home page')},
	{path: '/about', callback: () => console.log('About page')},
];


/**
 * @example
 * const router = new Router(routes);
 * router.navigateTo('/about');
 * 
 * window.addEventListener('popstate', () => {
 * 	router._loadInitialRoute();
 * });
 */
class Router {
	constructor(routes) {
		this.routes = routes;
		this._loadInitialRoute();
	}

	_getCurrentURL() {
		const path = window.location.pathname;
		return path;
	}

	_matchUrlToRoute(urlSegs) {
		const matchedRoute = this.routes.find(route => route.path === urlSegs);
		return matchedRoute;
	}

	_loadInitialRoute() {
		const pathnameSplit = window.location.pathname.split('/');
		const pathSegs = pathnameSplit.length > 1 ? pathnameSplit.slice(1) : '';

		this._loadInitialRoute(...pathSegs);
	}

	loadRoute(...urlSegs) {
		const matchedRoute = this._matchUrlToRoute(urlSegs);
		if (!matchedRoute) {
			throw new Error('Route not found!');
		}
		matchedRoute.callback();
	}

	navigateTo(path) {
		window.history.pushState({}, '', path);
		this.loadRoute(path);
	}
}