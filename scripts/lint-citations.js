const fs = require('node:fs');
const path = require('node:path');

const ROOT = process.cwd();
const LEDGER_DIR = path.join(ROOT, 'docs/evidence/source-ledger');
const SKIP_DIRS = new Set(['.git', '.vitepress', 'node_modules', 'generated']);
const citePattern = /\[CITE:\s*([a-z0-9][a-z0-9-]*)\]/g;
const problems = [];
const usedCards = new Map();

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    if (SKIP_DIRS.has(entry.name)) return [];
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) return walk(fullPath);
    return entry.isFile() && entry.name.endsWith('.md') ? [fullPath] : [];
  });
}

if (!fs.existsSync(LEDGER_DIR)) {
  console.error('citation lint failed: missing docs/evidence/source-ledger');
  process.exit(1);
}

for (const filePath of walk(ROOT)) {
  const relativePath = path.relative(ROOT, filePath);
  const lines = fs.readFileSync(filePath, 'utf8').split('\n');
  lines.forEach((line, index) => {
    for (const match of line.matchAll(citePattern)) {
      const slug = match[1];
      const cardPath = path.join(LEDGER_DIR, `${slug}.md`);
      if (!fs.existsSync(cardPath)) {
        problems.push(`${relativePath}:${index + 1}: missing source card "${slug}"`);
      } else {
        const locations = usedCards.get(slug) || [];
        locations.push(`${relativePath}:${index + 1}`);
        usedCards.set(slug, locations);
      }
    }
  });
}

for (const entry of fs.readdirSync(LEDGER_DIR)) {
  if (!entry.endsWith('.md') || entry === 'README.md') continue;
  const slug = entry.slice(0, -3);
  if (!usedCards.has(slug)) {
    problems.push(`docs/evidence/source-ledger/${entry}: source card is not cited`);
  }
}

if (problems.length > 0) {
  console.error('citation lint failed:');
  problems.forEach((problem) => console.error(`- ${problem}`));
  process.exit(1);
}

console.log(`citation lint passed for ${usedCards.size} source card(s)`);
