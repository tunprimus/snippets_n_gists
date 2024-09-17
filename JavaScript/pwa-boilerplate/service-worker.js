//@ts-check
'use strict';

const STATIC_CACHE_NAME = 'site-static-v2';
const DYNAMIC_CACHE_NAME = 'site-dynamic-v1';
const ONLINE_URL = 'https://www.example.com';

const ASSETS_TO_CACHE = [
	'./',
	'./index.html',
	'./js/main.js',
	'./js/utils/utils.js',
	'./css/style.css',
	'./pages/404.html',
	'./pages/offline.html',
];

/**
 * This function will limit the cache size to the given size.
 * @param {string} name - The name of the cache.
 * @param {number} size - The maximum size of the cache.
 */
function limitCacheSize(name, size) {
	caches.open(name).then((cache) => {
		cache.keys().then(keys => {
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
self.addEventListener('install', evt => {
	// console.log('service worker has been installed');
	evt.waitUntil(
		caches.open(STATIC_CACHE_NAME)
			.then((cache) => {
				console.log('caching shell assets');
				cache.addAll(ASSETS_TO_CACHE)
		}).then(() => self.skipWaiting())
	);
});


/**
 * Activate event
 * @param {*} evt
 */
self.addEventListener('activate', (evt) => {
	// console.log('service worker has been activated');
	evt.waitUntil(
		caches.keys().then(keys => {
			// console.log(keys);
			return Promise.all(keys
				.filter(key => key !== STATIC_CACHE_NAME && key !== DYNAMIC_CACHE_NAME)
				.map(key => caches.delete(key)));
		})
	);
});


/**
 * Fetch event
 * @param {*} evt
 */
self.addEventListener('fetch', (evt) => {
	if (evt.request.url.indexOf(ONLINE_URL) === -1) {
		evt.respondWith(
			caches.match(evt.request).then(cacheRes => {
				return cacheRes || fetch(evt.request).then((fetchRes) => {
					return caches.open(DYNAMIC_CACHE_NAME).then(cache => {
						cache.put(evt.request.url, fetchRes.clone());
						limitCacheSize(DYNAMIC_CACHE_NAME, 15);
						return fetchRes;
					});
				});
			}).catch(() => {
				if (evt.request.url.indexOf('.html') > -1) {
					return caches.match('./pages/fallback.html');
				}
			})
		);
	}
});
