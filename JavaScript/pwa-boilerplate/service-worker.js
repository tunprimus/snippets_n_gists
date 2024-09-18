//@ts-check
'use strict';

const RESOURCE_VERSION = '1';
const STATIC_CACHE_NAME = `PWA-name-static-v${RESOURCE_VERSION}`;
const DYNAMIC_CACHE_NAME = `PWA-name-dynamic-v${RESOURCE_VERSION}`;
const ONLINE_URL = 'https://www.example.com';

const ASSETS_TO_CACHE = [
	'./',
	'./index.html',
	'./js/main.js',
	'./js/utils/utils.js',
	'./css/style.css',
	'./pages/offline.html',
];

const ASSETS_TO_CACHE_WITH_VERSIONS = ASSETS_TO_CACHE.map(function(path) {
	return `${path}?v=${RESOURCE_VERSION}`;
})

/**
 * This function will limit the cache size to the given size.
 * @param {string} name - The name of the cache.
 * @param {number} size - The maximum size of the cache.
 */
function limitCacheSize(name, size) {
	return caches.open(name).then(function(cache) {
		return cache.keys().then(keys => {
			if(keys.length > size){
				cache.delete(keys[0]).then(limitCacheSize(name, size));
			}
		});
	});
};


/**
 * Install service worker
 * @param {*} evt
 */
self.addEventListener('install', function(evt) {
	 // Perform install step:  loading each required file into cache
	evt.waitUntil(
		caches.open(STATIC_CACHE_NAME)
			.then(function(cache) {
				// Add all offline dependencies to the cache
				console.log('caching shell assets');
				return cache.addAll(ASSETS_TO_CACHE_WITH_VERSIONS);
		}).then(function() {
			// At this point everything has been cached
			return self.skipWaiting();
		})
	);
});


/**
 * Activate event
 * @param {*} evt
 */
self.addEventListener('activate', function(evt) {
	return evt.waitUntil(
		caches.keys().then(function(keys) {
			return Promise.all(keys
				.filter(key => key !== STATIC_CACHE_NAME && key !== DYNAMIC_CACHE_NAME)
				.map(key => caches.delete(key)));
		}).then(function() {
			return self.clients.claim()
		})
	);
});


/**
 * Fetch event
 * @param {*} evt
 */
self.addEventListener('fetch', function(evt) {
	if (evt.request.url.indexOf(ONLINE_URL) === -1) {
		return evt.respondWith(
			caches.match(evt.request).then(function(cacheRes) {
				return cacheRes || fetch(evt.request).then(function(fetchRes) {
					return caches.open(DYNAMIC_CACHE_NAME).then(function(cache) {
						cache.put(evt.request.url, fetchRes.clone());
						limitCacheSize(DYNAMIC_CACHE_NAME, 15);
						return fetchRes;
					});
				});
			}).catch(function() {
				if (evt.request.url.indexOf('.html') > -1) {
					return caches.match('./pages/fallback.html');
				}
			})
		);
	}
});
