//@ts-check

{
	'use strict';

	/**
	 * Constant, helpers
	 */

	const View = {
		LOADING: 0,
		OK: 1,
		NEEDS_AUTH: 2,
		ERROR: 3,
	};

	/**
	 * Application state
	 */

	let state = {
		loading: true,
		authenticated: false,
		error: false,
		data: null,
	};

	/**
	 * State accessor functions (getters and setters)
	 */

	function loadData() {
		return fetch('/verify')
			.then(response => {
				state.authenticated = response.ok;
				if (response && response.ok) {
					return fetch('/data/');
				}
			})
			.then(response => {
				state.error = !response.ok;
				if (response && response.ok) {
					return response.json();
				}
			})
			.then(data => {
				state.data = (data !== null && data !== undefined) ? data : null;
				state.loading = false;
			})
			.catch(() => {
				state.loading = false;
				state.error = true;
				state.data = null;
			});
	}

	function currentView() {
		if (state.loading) {
			return View.LOADING;
		}
		if (state.error) {
			return View.ERROR;
		}
		if (!state.authenticated) {
			return View.NEEDS_AUTH;
		}
		return View.READY;
	}

	function messageCount() {
		return state.data && state.data.messages.length;
	}

	/**
	 * DOM node references
	 */
	const viewLoadingElem = document.getElementById('view-loading');
	const viewFailureElem = document.getElementById('view-failure');
	const viewNeedsLoginElem = document.getElementById('view-needs-login');
	const viewReadyElem = document.getElementById('view-ready');
	let currentViewElem = viewLoadingElem;
	// NB: indexes match View ID's
	let viewNodesReference = {
		viewLoadingElem,
		viewReadyElem,
		viewNeedsLoginElem,
		viewFailureElem,
	};
	const messageCountElem = document.getElementById('message-count');

	/**
	 * DOM update functions
	 */

	function updateView() {
		let nextViewElem = viewNodesReference[currentView()];
		if (nextViewElem === currentViewElem) {
			return;
		}
		currentViewElem.classList.add('hidden');
		nextViewElem.classList.remove('hidden');
		currentViewElem = nextViewElem;
	}

	function updateMessageCount() {
		let num = (messageCount() !== null && messageCount() !== undefined) ? messageCount() : 0;
		messageCountElem.textContent = `${num || 'no'} message${num == 1 ? '' : 's'}`;
		messageCountElem.classList.toggle('warning', (num > 100));
	}

	function updateInitialView() {
		updateMessageCount();
		updateView();
	}

	/**
	 * Event handlers
	 */

	/**
	 * Event handler bindings
	 */

	/**
	 * Initial setup
	 */
	loadData().then(updateInitialView);
}