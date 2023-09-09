/**
 * Inspired by Parsing Metadata Inside of Markdown Using JavaScript Without Any Dependencies #+++>>> https://ilikekillnerds.com/2023/01/parsing-metadata-inside-of-markdown-using-javascript-without-any-dependencies/
 */

const { re } = require("semver");

const parseMarkdownMetadata = markdown => {
  // Regular expression to match metadata at the beginning of the file
  const metadataRegex = /^---([\s\S]*?)---/;
  const metadataMatch = markdown.match(metadataRegex);

  // If there is no metadata, return empty object
  if (!metadataMatch) {
    return {};
  }

  // Split the metadata into lines
  const metadataLines = metadataMatch[1].split('\n');

  // Use reduce to accumulate the metadata as an object
  const metadata = metadataLines.reduce((acc, line) => {
    // Split the line into key-value pairs
    const [key, value] = line.split(':').map(part => part.trim());

    // If the line is not empty add the key-value pair to the metadata object
    if (key) {
      acc[key] = value;
    }
    return acc;
  }, {});

  // Return the metadata object
  return metadata;
};

/* Demo */
const markdown = `---
title: My Fictitious Blog Post
author: John Doe
date: 2023-09-09
---
# My Blog Post
This is the content of my fictitious blog post.`;

let metadata = parseMarkdownMetadata(markdown);
console.log(metadata);
