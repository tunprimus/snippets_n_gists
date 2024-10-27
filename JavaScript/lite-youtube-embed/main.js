//@ts-check

function createIframe(div) {
	let iframe = document.createElement('iframe');
	iframe.setAttribute('class', 'youtube-player__iframe');
	iframe.setAttribute('src', 'http://www.youtube.com/embed/' + div.dataset.id + '?autoplay=1');
	iframe.setAttribute('frameborder', '0');
	iframe.setAttribute('allowfullscreen', '1');
	iframe.setAttribute('allow', 'accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture');
	div.parentNode.replaceChild(iframe, div);
}

function initYouTubeVideos() {
	let playerElements = document.querySelectorAll('.youtube-player');
	const playerElementsLength = playerElements.length;
	for (let i = 0; i < playerElementsLength; i++) {
		let videoId = playerElements[i].dataset.id;
		let div = document.createElement('div');
		div.setAttribute('data-id', videoId);
		let thumbNode = document.createElement('img');
		thumbNode.src = '//i.ytimg.com/vi/ID/hqdefault.jpg'.replace('ID', videoId);
		thumbNode.setAttribute('class', 'youtube-player__img');
		div.appendChild(thumbNode);
		let playButton = document.createElement('div');
		playButton.setAttribute('class', 'youtube-player__play');
		div.appendChild(playButton);
		div.onclick = function() {
			createIframe(this);
		};
		playerElements[i].appendChild(div);
	}
}

document.addEventListener('DOMContentLoaded', initYouTubeVideos);
