/**
 * html fetch multiple files
 * @link https://stackoverflow.com/questions/29676408/html-fetch-multiple-files
 * 
 * 
  1.  Loop through all URLs.
  2.  For each URL, fetch it with the fetch API, store the returned promise in list.
  3.  Additionally, when the request is finished, store the result in results.
  4.  Create a new promise, that resolves, when all promises in list are resolved (i.e., all requests finished).
  5.  Enjoy the fully populated results!
 */

var list = [];
var urls = ['1.html', '2.html', '3.html',];
var results = [];

urls.forEach(function (url, i) {
  // (1)
  list.push(
    // (2)
    fetch(url)
      .then(function (res) {
        results[i] = res.blob(); // (3)
      })
  );
});

Promise.all(list) // (4)
  .then(function () {
    alert("all requests finished!"); // (5)
  });


function fetchAll(...resources) {
  var destination = [];
  resources.forEach((it) => {
    destination.push(fetch(it));
  });
  return Promise.all(destination);
}
