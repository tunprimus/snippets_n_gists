// @ts-check
/**
 * @description Native conversion of CSV to JSON
 * @author Tari Ibaba
 * @link https://www.codingbeautydev.com/blog/javascript-convert-csv-to-json
 */

/**
 * @function csvToJson function to convert CSV to JSON
 * @param {string} csv string to parse
 * @returns 
 */
function csvToJson(csv) {
  const lines = csv.split('\n');
  const delimiter = ',';

  const result = [];

  const headers = lines[0].split(delimiter);

  for (const line of lines) {
    const obj = {};
    const row = line.split(delimiter);

    for (let  i = 0; i < headers.length; i++) {
      const header = headers[i];
      obj[header] = row[i];
    }

    result.push(obj);
  }

  return result;
}

export { csvToJson };
