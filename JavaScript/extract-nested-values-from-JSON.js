//@ts-check

const googleJSON = {
"destination_addresses": [
  "Washington, DC, USA",
  "Philadelphia, PA, USA",
  "Santa Barbara, CA, USA",
  "Miami, FL, USA",
  "Austin, TX, USA",
  "Napa County, CA, USA"
],
"origin_addresses": [
  "New York, NY, USA"
],
"rows": [{
  "elements": [{
      "distance": {
        "text": "227 mi",
        "value": 365468
      },
      "duration": {
        "text": "3 hours 54 mins",
        "value": 14064
      },
      "status": "OK"
    },
    {
      "distance": {
        "text": "94.6 mi",
        "value": 152193
      },
      "duration": {
        "text": "1 hour 44 mins",
        "value": 6227
      },
      "status": "OK"
    },
    {
      "distance": {
        "text": "2,878 mi",
        "value": 4632197
      },
      "duration": {
        "text": "1 day 18 hours",
        "value": 151772
      },
      "status": "OK"
    },
    {
      "distance": {
        "text": "1,286 mi",
        "value": 2069031
      },
      "duration": {
        "text": "18 hours 43 mins",
        "value": 67405
      },
      "status": "OK"
    },
    {
      "distance": {
        "text": "1,742 mi",
        "value": 2802972
      },
      "duration": {
        "text": "1 day 2 hours",
        "value": 93070
      },
      "status": "OK"
    },
    {
      "distance": {
        "text": "2,871 mi",
        "value": 4620514
      },
      "duration": {
        "text": "1 day 18 hours",
        "value": 152913
      },
      "status": "OK"
    }
  ]
}],
"status": "OK"
};

/**
 * To get the values of a key, no matter the level of nesting. Converted from Python to JavaScript
 * @author Todd Birchard
 * @link https://gist.github.com/toddbirchard/b6f86f03f6cf4fc9492ad4349ee7ff8b
 * @param {object} json - Nested JSON to search
 * @param {string} key - Key to search for
 * @returns {array}
 */
function jsonExtract(json, key) {
	var arr = [];
	const obj = JSON.parse(json);

	function extract(obj, arr, key) {
		if (typeof obj === 'object') {
			for (var k in obj) {
				if (obj.hasOwnProperty(k)) {
					var v = obj[k];
					if (typeof v === 'object') {
						extract(v, arr, key);
					} else if (k === 'key') {
						arr.push(v);
					}
				}
			}
		} else if (Array.isArray(obj)) {
			for (var i = 0; i < obj.length; i++) {
				extract(obj[i], arr, key);
			}
		}
		return arr;
	}
	var values = extract(obj, arr, key);
	return values;
}

console.log(jsonExtract(googleJSON, 'text'));
