//@ts-check
'use strict';



/**
 * Register service worker
 */
if ('serviceWorker' in navigator) {
	navigator.serviceWorker.register('./service-worker.js', {scope: '.'})
		.then((reg) => console.log('service worker registered', reg))
		.catch((err) => console.error('service worker error not registered', err));
}
