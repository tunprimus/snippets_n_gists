//@ts-check
'use strict';
/* Adapted from: https://gist.github.com/cferdinandi/6e4a73a69b0ee30c158c8dd37d314663 */

let coreAssets = [];

// On install, cache core assets
self.addEventListener('install', function (evt) {
	// Cache core assets
	evt.waitUntil(caches.open('app').then(function (cache) {
		for (let asset of coreAssets) {
			cache.add(new Request(asset));
		}
		return cache;
	}))
});

// Listen for request events
self.addEventListener('fetch', function (evt) {
	// Get the request
	let request = evt.request;

	// Bug fix for "What causes a Failed to execute 'fetch' on 'ServiceWorkerGlobalScope': 'only-if-cached' can be set only with 'same-origin' mode error?"
	// https://stackoverflow.com/a/49719964
	if (evt.request.cache === 'only-if-cached' && evt.request.mode !== 'same-origin') {
		return;
	}

	// HTML files
	// Network-first
	if (request.headers.get('Accept').includes('text/html')) {
		evt.respondWith(
			fetch(request).then(function (response) {
				// Create a copy of the response and save it to the cache
				let respCopy = response.clone();
				evt.waitUntil(caches.open('app').then(function (cache) {
					return cache.put(request, respCopy);
				}));
				// Return the response
				return response;
			}).catch(function (err) {
				// If there is no item in cache, respond with a fallback
				return caches.match(request).then(function (response) {
					return response || caches.match('/offline.html');
				});
			})
		);
	}

	// CSS & JavaScript
	// Offline-first
	if (request.headers.get('Accept').includes('text/css') || request.headers.get('Accept').includes('text/javascript')) {
		evt.respondWith(
			caches.match(request).then(function (response) {
				return response || fetch(request).then(function (response) {
					// Return the response
					return response;
				})
			})
		);
		return;
	}

	// Images
	// Offline-first
	if (request.headers.get('Accept').includes('image')) {
		evt.respondWith(
			caches.match(request).then(function (response) {
				return response || fetch(request).then(function (response) {
					// Save a copy of it in the cache
					let respCopy = response.clone();
					evt.waitUntil(caches.open('app').then(function (cache) {
						return cache.put(request, respCopy);
					}));
					// Return the response
					return response;
				});
			})
		);
	}
});
