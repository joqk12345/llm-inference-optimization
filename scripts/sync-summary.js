const {
  chapterNumberFromTitle,
  parseFrontmatter,
  stringifyFrontmatter,
  writeText,
  loadText
} = require('./knowledge-lib.js');

const raw = loadText('SUMMARY.md');
const parsed = parseFrontmatter(raw);
const lines = parsed.content.split('\n');
let currentSection = null;
let changed = false;

function finalizeSection() {
  if (!currentSection) return;
  const numbers = currentSection.entries
    .map((entry) => chapterNumberFromTitle(entry))
    .filter((value) => value !== null);

  if (numbers.length === 0) return;

  const min = Math.min(...numbers);
  const max = Math.max(...numbers);
  const suffix = min === max ? `（第${min}章）` : `（第${min}-${max}章）`;
  const normalizedTitle = currentSection.title.replace(/（第\d+(?:-\d+)?章）$/, '').trim();
  const expected = `${normalizedTitle}${suffix}`;
  if (expected !== currentSection.title) {
    lines[currentSection.index] = `*   **${expected}**`;
    changed = true;
  }
}

lines.forEach((line, index) => {
  const sectionMatch = line.match(/^\*\s+\*\*(.+?)\*\*\s*$/);
  if (sectionMatch) {
    finalizeSection();
    currentSection = { index, title: sectionMatch[1].trim(), entries: [] };
    return;
  }
  const entryMatch = line.match(/^\s+\*\s+\[([^\]]+)\]\(([^)]+)\)\s*$/);
  if (entryMatch && currentSection) {
    currentSection.entries.push(entryMatch[1].trim());
  }
});

finalizeSection();

if (changed) {
  writeText('SUMMARY.md', stringifyFrontmatter(parsed.data, lines.join('\n')));
  console.log('SUMMARY.md synchronized');
} else {
  console.log('SUMMARY.md already synchronized');
}
