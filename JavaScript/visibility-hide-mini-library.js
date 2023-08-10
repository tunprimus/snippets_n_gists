/* Inspired by Onion - https://www.npmjs.com/package/onion?activeTab=readme */

function setClasses(element, classes) {
  Object.entries(classes)
    .filter(([key]) => key.length)
    .forEach(([key, value]) => element.classList.toggle(key, value));
}

function addEndListener (element, listener) {
  element.addEventListener('transitionend', listener);
  element.addEventListener('animationend', listener);
}

function removeEndListener (element, listener) {
  element.removeEventListener('transitionend', listener);
  element.removeEventListener('animationend', listener);
}

function showElement (element, token = '') {
  if (!(element instanceof HTMLElement)) {
    return;
  }

  if (element.classList.contains('is-opening')) {
    return;
  }
  /* For later */
}
/* For later */