const { getCoreSummarySections, writeText } = require('./knowledge-lib.js');

const lines = [
  '# Generated Summary',
  '',
  '> Auto-generated from `SUMMARY.md`. This page is the single machine-built reading index for the core book path.',
  ''
];

getCoreSummarySections().forEach((section) => {
  lines.push(`## ${section.title}`);
  section.entries.forEach((entry) => {
    lines.push(`- [${entry.title}](/${entry.filePath.replace(/\.md$/, '')})`);
  });
  lines.push('');
});

writeText('generated/summary.md', lines.join('\n'));
console.log('generated/summary.md created');
