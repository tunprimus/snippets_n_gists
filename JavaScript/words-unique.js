//@ts-check

/**
 * @description For the given `text` in the language `lang`, return the set of unique words.
 * @link https://danburzo.ro/snippets/words-unique/
 * @param {object | string[] | string} text 
 * @param {string} lang 
 * @returns {Set<string>}
 */
function wordsUnique(text, lang = 'en') {
	return new Set(
		Array.from(new Intl.Segmenter(lang, { granularity: 'word' }).segment(text))
			.filter(i => i.isWordLike)
			.map(i => i.segment)
	);
}

export { wordsUnique };
