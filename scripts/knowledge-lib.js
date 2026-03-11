const fs = require('node:fs');
const path = require('node:path');
const { PRESET_METADATA, SECONDARY_FILES } = require('./knowledge-presets.js');

const SUMMARY_PATH = 'SUMMARY.md';
const FRONTMATTER_REQUIRED_FIELDS = [
  'id',
  'title',
  'slug',
  'date',
  'type',
  'topics',
  'concepts',
  'tools',
  'architecture_layer',
  'learning_stage',
  'optimization_axes',
  'related',
  'references',
  'status',
  'display_order'
];

const DEFAULT_DATE = '2026-03-11';
const SUMMARY_SECTION_LINE_PATTERN = /^\*\s+\*\*(.+?)\*\*\s*$/;
const SUMMARY_ENTRY_LINE_PATTERN = /^(\s*)\*\s+\[([^\]]+)\]\(([^)]+\.md)\)\s*$/;

function normalizePath(filePath) {
  return filePath.replace(/\\/g, '/').replace(/^\.\//, '');
}

function parseScalar(value) {
  const trimmed = value.trim();
  if (!trimmed) return '';
  if (trimmed === '[]') return [];
  if (/^\d+$/.test(trimmed)) return Number(trimmed);
  if (trimmed === 'true') return true;
  if (trimmed === 'false') return false;
  if (trimmed.startsWith('"') && trimmed.endsWith('"')) return JSON.parse(trimmed);
  if (trimmed.startsWith("'") && trimmed.endsWith("'")) return trimmed.slice(1, -1);
  return trimmed;
}

function quoteScalar(value) {
  if (typeof value === 'number') return String(value);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return JSON.stringify(String(value));
}

function parseFrontmatter(raw) {
  if (!raw.startsWith('---\n')) return { data: {}, content: raw, hasFrontmatter: false };
  const end = raw.indexOf('\n---\n', 4);
  if (end === -1) return { data: {}, content: raw, hasFrontmatter: false };
  const fm = raw.slice(4, end);
  const content = raw.slice(end + 5);
  const data = {};
  let currentArrayKey = null;

  fm.split('\n').forEach((line) => {
    if (!line.trim()) return;
    if (line.startsWith('  - ') && currentArrayKey) {
      data[currentArrayKey].push(parseScalar(line.slice(4)));
      return;
    }
    const idx = line.indexOf(':');
    if (idx === -1) return;
    const key = line.slice(0, idx).trim();
    const value = line.slice(idx + 1).trim();
    if (!value) {
      data[key] = [];
      currentArrayKey = key;
      return;
    }
    data[key] = parseScalar(value);
    currentArrayKey = null;
  });

  return { data, content, hasFrontmatter: true };
}

function stringifyFrontmatter(data, content) {
  const lines = [];
  Object.entries(data).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      if (value.length === 0) {
        lines.push(`${key}: []`);
      } else {
        lines.push(`${key}:`);
        value.forEach((item) => lines.push(`  - ${quoteScalar(item)}`));
      }
      return;
    }
    lines.push(`${key}: ${quoteScalar(value)}`);
  });
  const normalizedContent = String(content || '').replace(/^\n+/, '');
  return `---\n${lines.join('\n')}\n---\n${normalizedContent}`;
}

function loadText(filePath) {
  return fs.readFileSync(filePath, 'utf8');
}

function writeText(filePath, content) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content.endsWith('\n') ? content : `${content}\n`);
}

function filePathToId(filePath) {
  return normalizePath(filePath)
    .replace(/\.md$/, '')
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();
}

function extractHeading(content, fallback) {
  const heading = content
    .split('\n')
    .find((line) => line.startsWith('# ') && !line.startsWith('#!'));
  return heading ? heading.slice(2).trim() : fallback;
}

function inferType(filePath) {
  if (filePath.startsWith('docs/cases/')) return 'case-study';
  if (filePath.startsWith('docs/')) return 'reference';
  if (filePath.startsWith('appendix-')) return 'reference';
  return 'article';
}

function ensureArray(value) {
  if (Array.isArray(value)) return value;
  if (value === undefined || value === null || value === '') return [];
  return [value];
}

function getSummarySections() {
  const raw = loadText(SUMMARY_PATH);
  const { content } = parseFrontmatter(raw);
  const sections = [];
  let currentSection = null;
  let order = 0;

  const ensureSection = (title) => {
    const existing = sections.find((section) => section.title === title);
    if (existing) return existing;
    const section = { title, entries: [] };
    sections.push(section);
    return section;
  };

  content.split('\n').forEach((line) => {
    const sectionMatch = line.match(SUMMARY_SECTION_LINE_PATTERN);
    if (sectionMatch) {
      currentSection = ensureSection(sectionMatch[1].trim());
      return;
    }

    const linkMatch = line.match(SUMMARY_ENTRY_LINE_PATTERN);
    if (!linkMatch) return;
    const [, indentation, title, target] = linkMatch;
    if (!currentSection || indentation.length === 0) {
      currentSection = ensureSection('前言');
    }
    order += 1;
    currentSection.entries.push({
      order,
      title: title.trim(),
      filePath: normalizePath(target),
      sectionTitle: currentSection.title
    });
  });

  return sections.filter((section) => section.entries.length > 0);
}

function getCoreSummarySections() {
  return getSummarySections();
}

function getCoreSummaryEntries() {
  return getCoreSummarySections().flatMap((section) => section.entries);
}

function getSecondaryFileEntries() {
  return SECONDARY_FILES.filter((filePath) => fs.existsSync(filePath)).map((filePath, index) => ({
    filePath,
    order: 200 + index
  }));
}

function getManagedContentFiles() {
  return [
    ...getCoreSummaryEntries().map((entry) => entry.filePath),
    ...getSecondaryFileEntries().map((entry) => entry.filePath)
  ];
}

function isCoreKnowledgeFile(filePath) {
  return getCoreSummaryEntries().some((entry) => entry.filePath === normalizePath(filePath));
}

function getPreset(filePath) {
  return PRESET_METADATA[normalizePath(filePath)] || {};
}

function deriveMetadata(filePath, order) {
  const normalizedPath = normalizePath(filePath);
  const raw = loadText(normalizedPath);
  const parsed = parseFrontmatter(raw);
  const preset = getPreset(normalizedPath);
  const title = preset.title || parsed.data.title || extractHeading(parsed.content || raw, path.basename(normalizedPath, '.md'));
  const id = preset.id || parsed.data.id || filePathToId(normalizedPath);
  const metadata = {
    id,
    title,
    slug: preset.slug || parsed.data.slug || id,
    date: preset.date || parsed.data.date || DEFAULT_DATE,
    type: preset.type || parsed.data.type || inferType(normalizedPath),
    topics: ensureArray(preset.topics || parsed.data.topics),
    concepts: ensureArray(preset.concepts || parsed.data.concepts),
    tools: ensureArray(preset.tools || parsed.data.tools),
    architecture_layer: ensureArray(preset.architecture_layer || parsed.data.architecture_layer),
    learning_stage: preset.learning_stage || parsed.data.learning_stage || 'orientation',
    optimization_axes: ensureArray(preset.optimization_axes || parsed.data.optimization_axes),
    related: ensureArray(preset.related || parsed.data.related),
    references: ensureArray(preset.references || parsed.data.references),
    status: preset.status || parsed.data.status || 'published',
    display_order: order ?? preset.display_order ?? parsed.data.display_order ?? 999
  };
  return metadata;
}

function parseArticle(filePath, order) {
  const normalizedPath = normalizePath(filePath);
  const raw = loadText(normalizedPath);
  const parsed = parseFrontmatter(raw);
  const metadata = deriveMetadata(normalizedPath, order);
  return {
    filePath: normalizedPath,
    raw,
    metadata,
    content: parsed.hasFrontmatter ? parsed.content : raw,
    hasFrontmatter: parsed.hasFrontmatter
  };
}

function parseSimpleYaml(filePath) {
  const content = loadText(filePath);
  const root = {};
  let currentKey = null;
  let currentObject = null;

  content.split('\n').forEach((line) => {
    if (!line.trim() || line.trim().startsWith('#')) return;
    if (/^[A-Za-z0-9_-]+:\s*$/.test(line)) {
      currentKey = line.slice(0, line.indexOf(':')).trim();
      root[currentKey] = [];
      currentObject = null;
      return;
    }
    if (!currentKey) return;
    const arrayItemMatch = line.match(/^\s+-\s+(.*)$/);
    if (arrayItemMatch) {
      const value = arrayItemMatch[1];
      if (value.includes(':')) {
        const idx = value.indexOf(':');
        currentObject = { [value.slice(0, idx).trim()]: parseScalar(value.slice(idx + 1).trim()) };
        root[currentKey].push(currentObject);
      } else {
        currentObject = null;
        root[currentKey].push(parseScalar(value));
      }
      return;
    }
    const objectPropMatch = line.match(/^\s+([A-Za-z0-9_-]+):\s*(.*)$/);
    if (objectPropMatch && currentObject) {
      currentObject[objectPropMatch[1]] = parseScalar(objectPropMatch[2]);
    }
  });

  return root;
}

function loadControlPlane() {
  return {
    taxonomy: parseSimpleYaml('meta/taxonomy.yaml'),
    architecture: parseSimpleYaml('meta/architecture-layers.yaml'),
    learningStages: parseSimpleYaml('meta/learning-stages.yaml'),
    optimizationAxes: parseSimpleYaml('meta/optimization-axes.yaml')
  };
}

function escapeTableCell(value) {
  return String(value).replace(/\|/g, '\\|').replace(/\n/g, '<br>');
}

function formatMarkdownTable(headers, rows) {
  return [
    `| ${headers.map(escapeTableCell).join(' | ')} |`,
    `| ${headers.map(() => '---').join(' | ')} |`,
    ...rows.map((row) => `| ${row.map(escapeTableCell).join(' | ')} |`)
  ];
}

function replaceAutoGeneratedBlock(filePath, token, lines) {
  const start = `<!-- AUTO-GENERATED:${token}:START -->`;
  const end = `<!-- AUTO-GENERATED:${token}:END -->`;
  const raw = loadText(filePath);
  const replacement = `${start}\n${lines.join('\n')}\n${end}`;
  const pattern = new RegExp(`${escapeRegExp(start)}[\\s\\S]*?${escapeRegExp(end)}`);
  const next = pattern.test(raw) ? raw.replace(pattern, replacement) : `${raw.trimEnd()}\n\n${replacement}\n`;
  writeText(filePath, next);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function chapterNumberFromTitle(title) {
  const match = String(title).match(/^第(\d+)章/);
  return match ? Number(match[1]) : null;
}

function chapterRefFromTitle(title) {
  const normalized = String(title).trim();
  const chapterMatch = normalized.match(/^(第\d+章)/);
  if (chapterMatch) return chapterMatch[1];
  const appendixMatch = normalized.match(/^(附录[A-Z])/);
  if (appendixMatch) return appendixMatch[1];
  return '前言';
}

function discoverMarkdownLinks(filePath, content) {
  const normalizedPath = normalizePath(filePath);
  const links = [];
  const pattern = /\[[^\]]*]\(([^)]+)\)/g;
  let match = pattern.exec(content);
  while (match) {
    const target = match[1].trim();
    links.push(resolveLinkTarget(normalizedPath, target));
    match = pattern.exec(content);
  }
  return links.filter(Boolean);
}

function resolveLinkTarget(sourcePath, target) {
  if (!target || target.startsWith('#') || target.startsWith('mailto:')) return null;
  if (/^https?:\/\//.test(target)) {
    return { kind: 'external', target };
  }
  const cleanTarget = target.split('#')[0];
  if (!cleanTarget.endsWith('.md')) return null;
  const resolved = normalizePath(path.join(path.dirname(sourcePath), cleanTarget));
  return { kind: 'internal', target: resolved };
}

function unique(values) {
  return Array.from(new Set(values));
}

module.exports = {
  DEFAULT_DATE,
  FRONTMATTER_REQUIRED_FIELDS,
  SECONDARY_FILES,
  chapterNumberFromTitle,
  chapterRefFromTitle,
  deriveMetadata,
  discoverMarkdownLinks,
  escapeRegExp,
  filePathToId,
  formatMarkdownTable,
  getCoreSummaryEntries,
  getCoreSummarySections,
  getManagedContentFiles,
  getSecondaryFileEntries,
  isCoreKnowledgeFile,
  loadControlPlane,
  loadText,
  normalizePath,
  parseArticle,
  parseFrontmatter,
  stringifyFrontmatter,
  replaceAutoGeneratedBlock,
  unique,
  writeText
};
