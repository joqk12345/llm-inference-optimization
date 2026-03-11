const { getCoreSummarySections, writeText } = require('./knowledge-lib.js');

const sections = getCoreSummarySections().map((section) => ({
  text: section.title,
  items: section.entries.map((entry) => ({
    text: entry.title,
    link: `/${entry.filePath.replace(/\.md$/, '')}`
  }))
}));

writeText('.vitepress/knowledge-nav.json', JSON.stringify(sections, null, 2));
console.log(`.vitepress/knowledge-nav.json created with ${sections.length} section(s)`);
