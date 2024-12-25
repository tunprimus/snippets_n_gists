//@ts-check
'use strict';

let posts = document.querySelectorAll('.target target');

let postsObj = [...posts].map(function (post) {
	return {
		title: post.querySelector('h1').innerText,
		subtitle: post.querySelector('h2').innerText,
		date: post.querySelector('time').innerText,
		url: post.querySelector('a').href,
		content: post.querySelector('p').innerText,
	};
});
