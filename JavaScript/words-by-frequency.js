//@ts-check

/**
 * @description For the given `text` in the language `lang`, 	return the set of unique words 	along with the number of occurrences, normalised to lowercase and sorted by frequency.
 * @link https://danburzo.ro/snippets/words-by-frequency/
 * @param {object | string[] | string} text 
 * @param {string} lang 
 * @returns {object}
 */
function wordsByFrequency(text, lang = 'en') {
	return Object.entries(
		Array.from(new Intl.Segmenter(lang, { granularity: 'word' }).segment(text))
			.filter(i => i.isWordLike)
			.map(i => i.segment.toLowerCase())
			.reduce((s, i) => (s[i] = (s[i] ?? 0) + 1, s), {})
	).sort((a, b) => b[1] - a[1]);
}

export { wordsByFrequency };
