const {
  getCoreSummaryEntries,
  getSecondaryFileEntries,
  parseArticle,
  stringifyFrontmatter,
  writeText
} = require('./knowledge-lib.js');

const entries = [
  ...getCoreSummaryEntries().map((entry) => ({ filePath: entry.filePath, order: entry.order })),
  ...getSecondaryFileEntries()
];

entries.forEach(({ filePath, order }) => {
  const article = parseArticle(filePath, order);
  const nextContent = stringifyFrontmatter(article.metadata, article.content);
  if (article.raw !== nextContent) {
    writeText(filePath, nextContent);
    console.log(`updated ${filePath}`);
  }
});

console.log(`ingested ${entries.length} managed file(s)`);
