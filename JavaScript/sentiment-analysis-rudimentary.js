/**
 * Inspired by Sentiment Analysis Using TypeScript Without Dependencies #+++>>> https://ilikekillnerds.com/2023/02/sentiment-analysis-using-typescript-without-dependencies/
 */

// Class to perform sentiment analysis on a given text
class SentimentAnalysis {
  // Arrays of positive and negative words to use in the analysis
  positiveWords = ['love', 'like', 'great', 'good', 'happy', 'awesome',];
  negativeWords = ['hate', 'dislike', 'bad', 'angry', 'sad', 'terrible',];

  // Method to perform sentiment analysis on a given text
  getSentiment(text) {
    // Convert the text to lowercase to make the analysis case-insensitive
    const lowerText = text.toLowerCase();

    // Sum up the number of times each positive word appears in the text
    let positiveScore = this.positiveWords.reduce((acc, word) => {
      // Use a regular expression to match the word in the text
      return acc + (lowerText.match(new RegExp(word, 'g')) || []).length;
    }, 0);

    // Sum up the number of times each negative word appears in the text
    let negativeScore = this.negativeWords.reduce((acc, word) => {
      // Use a regular expression to match the word in the text
      return acc + (lowerText.match(new RegExp(word, 'g')) || []).length;
    }, 0);

    // Calculate the number of neutral words by subtracting the positive and negative words from the total number of words
    let neutralScore = lowerText.split(' ').length - positiveScore - negativeScore;

    // Compare the number of positive and negative words and return the sentiment result
    if (positiveScore > negativeScore) {
      return {
        sentiment: 'positive',
        positiveWords: positiveScore,
        negativeWords: negativeScore,
        neutralWords: neutralScore,
      }
    } else if (positiveScore < negativeScore) {
      return {
        sentiment: 'negative',
        positiveWords: positiveScore,
        negativeWords: negativeScore,
        neutralWords: neutralScore,
      }
    } else {
      return {
        sentiment: 'neutral',
        positiveWords: positiveScore,
        negativeWords: negativeScore,
        neutralWords: neutralScore,
      }
    }
  }
}

/* Demo */

// Create an instance of the SentimentAnalysis class
const sentimentAnalysis = new SentimentAnalysis();

// Analyse some sample text
const result = sentimentAnalysis.getSentiment(
  'I love this code and think it is great!'
);

// Log the result
console.log(result);
