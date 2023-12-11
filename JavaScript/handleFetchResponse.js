
/**
 * How to handle response .json and .text using fetch?
 * @link https://stackoverflow.com/a/56227515
 */


/**
 * 
 * @param {object} response 
 * @returns 
 */
function handleResponseStatusAndContentType(response) {
  const contentType = response.headers.get('content-type');

  if (response.status === 401) {
    throw new Error('Request was not authorised.');
  }

  if (contentType === null) {
    return Promise.resolve(null);
  } else if (contentType.startsWith('application/json;')) {
    return response.json();
  } else if (contentType.startsWith('text/plain;')) {
    return response.text();
  } else {
    throw new Error(`Unsupported response content-type: ${contentType}`);
  }
}

function handleFetchResponse(address, options) {
  const url = address;
  const requestInit = options;
  
  return fetch(
    url,
    requestInit,
  )
  .then(response => handleResponseStatusAndContentType(response))
  .catch(error => {
    console.error(error);
    return error;
  });
}

export { handleFetchResponse };
