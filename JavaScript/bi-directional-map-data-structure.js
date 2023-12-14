/**
 * @description How I build My Own Data structure for Indexing : Bidirectional Map
 * @author Ashish maurya
 * @link https://blog.theashishmaurya.me/how-i-build-my-own-data-structure-for-indexing-bidirectional-map
 * @template K - The type of keys.
 * @template V - The type of values.
 *
 * @class
 * @classdesc A simple bidirectional map for mapping between keys and values.
 *   Provides methods to add mappings and retrieve values by key or value.
 *
 * @example
 * // Example usage:
 * const bidirectionalMap = new BidirectionalMap<string, number>();
 * bidirectionalMap.addMapping('apple', 1);
 * bidirectionalMap.addMapping('orange', 2);
 *
 * // Using key to get value
 * console.log(bidirectionalMap.getValue('apple')); // Output: 1
 *
 * // Using value to get key
 * console.log(bidirectionalMap.getValue(2)); // Output: orange
 */
class BidirectionalMap {
  /**
   * Creates a new instance of BidirectionalMap.
   * @constructor
   */
  constructor() {
    this.forwardMap = new Map(); // used to map key ===> path
    this.reverseMap = new Map(); // used to map path ====> key
  }

  /**
   * Adds a bidirectional mapping between a key and a value.
   *
   * @param {K} key - The key to be mapped.
   * @param {V} value - The value to be mapped.
   * @returns {void}
   * @memberof BidirectionalMap
   */
  addMapping(key, value) {
    this.forwardMap.set(key, value);
    this.reverseMap.set(value, key);
  }

  /**
   * Retrieves the value associated with the provided key or value.
   *
   * @param {K | V} keyOrValue - The key or value for which to retrieve the associated value or key.
   * @returns {V | K | undefined} The associated value or key, or undefined if not found.
   * @memberof BidirectionalMap
   */
  getValue(keyOrValue) {
    if (this.forwardMap.has(keyOrValue)) {
      return this.forwardMap.get(keyOrValue);
    } else if (this.reverseMap.has(keyOrValue)) {
      return this.reverseMap.get(keyOrValue);
    } else {
      return undefined;
    }
  }

  getAllKeysFromForwardMap() {
    return this.forwardMap.keys();
  }

  getAllKeysFromReverseMap() {
    return this.reverseMap.keys();
  }

  getAllValuesFromForwardMap() {
    return this.forwardMap.values();
  }

  getAllValuesFromReverseMap() {
    return this.reverseMap.values();
  }

  getValueFromForwardMap(key) {
    return this.forwardMap.get(key);
  }

  getValueFromReverseMap(key) {
    return this.reverseMap.get(key);
  }

  // TODO: This should update the the values in the both things

  updateValue(key, value) {
    // Update the value of key in forward Map and backward Map
    this.addMapping(key, value);
  }

  getMapping() {
    return {
      forwardMap: this.forwardMap,
      backwardMap: this.reverseMap
    }
  }
}

// Example usage:
const bidirectionalMap = new BidirectionalMap();
bidirectionalMap.addMapping("apple", 1);
bidirectionalMap.addMapping("orange", 2);

// Using key to get value
console.log(bidirectionalMap.getValue("apple")); // Output: 1

// Using value to get key
console.log(bidirectionalMap.getValue(2)); // Output: orange
