const {
  FRONTMATTER_REQUIRED_FIELDS,
  getCoreSummaryEntries,
  getManagedContentFiles,
  loadControlPlane,
  parseArticle
} = require('./knowledge-lib.js');

const control = loadControlPlane();
const managedFiles = getManagedContentFiles();
const problems = [];
const ids = new Map();
const nodes = [];

managedFiles.forEach((filePath, index) => {
  const article = parseArticle(filePath, index + 1);
  nodes.push(article);

  if (!article.hasFrontmatter) {
    problems.push(`${filePath}: missing frontmatter`);
  }

  FRONTMATTER_REQUIRED_FIELDS.forEach((field) => {
    const value = article.metadata[field];
    const missing = Array.isArray(value) ? value === undefined : value === undefined || value === '';
    if (missing) problems.push(`${filePath}: missing required field "${field}"`);
  });

  if (ids.has(article.metadata.id)) {
    problems.push(`${filePath}: duplicate id "${article.metadata.id}" with ${ids.get(article.metadata.id)}`);
  } else {
    ids.set(article.metadata.id, filePath);
  }
});

const allowed = {
  types: new Set(control.taxonomy.types || []),
  topics: new Set(control.taxonomy.topics || []),
  concepts: new Set(control.taxonomy.concepts || []),
  tools: new Set(control.taxonomy.tools || []),
  status: new Set(control.taxonomy.status || []),
  architectureLayers: new Set((control.architecture.layers || []).map((item) => item.name)),
  learningStages: new Set((control.learningStages.stages || []).map((item) => item.name)),
  optimizationAxes: new Set((control.optimizationAxes.axes || []).map((item) => item.name))
};

nodes.forEach((article) => {
  const { filePath, metadata } = article;
  if (!allowed.types.has(metadata.type)) problems.push(`${filePath}: invalid type "${metadata.type}"`);
  if (!allowed.status.has(metadata.status)) problems.push(`${filePath}: invalid status "${metadata.status}"`);
  if (!allowed.learningStages.has(metadata.learning_stage)) {
    problems.push(`${filePath}: invalid learning_stage "${metadata.learning_stage}"`);
  }
  metadata.topics.forEach((topic) => {
    if (!allowed.topics.has(topic)) problems.push(`${filePath}: unknown topic "${topic}"`);
  });
  metadata.concepts.forEach((concept) => {
    if (!allowed.concepts.has(concept)) problems.push(`${filePath}: unknown concept "${concept}"`);
  });
  metadata.tools.forEach((tool) => {
    if (!allowed.tools.has(tool)) problems.push(`${filePath}: unknown tool "${tool}"`);
  });
  metadata.architecture_layer.forEach((layer) => {
    if (!allowed.architectureLayers.has(layer)) problems.push(`${filePath}: unknown architecture layer "${layer}"`);
  });
  metadata.optimization_axes.forEach((axis) => {
    if (!allowed.optimizationAxes.has(axis)) problems.push(`${filePath}: unknown optimization axis "${axis}"`);
  });
});

nodes.forEach((article) => {
  const { filePath, metadata } = article;
  metadata.related.forEach((target) => {
    if (!ids.has(target)) problems.push(`${filePath}: related target "${target}" not found`);
  });
  metadata.references.forEach((target) => {
    if (!/^https?:\/\//.test(target) && !ids.has(target)) {
      problems.push(`${filePath}: reference target "${target}" not found`);
    }
  });
});

const summaryEntries = getCoreSummaryEntries();
summaryEntries.forEach((entry) => {
  if (!managedFiles.includes(entry.filePath)) {
    problems.push(`SUMMARY.md references unmanaged file "${entry.filePath}"`);
  }
});

if (summaryEntries.length !== 15) {
  problems.push(`SUMMARY.md should contain 15 core reading-path entries, found ${summaryEntries.length}`);
}

if (problems.length > 0) {
  console.error('knowledge lint failed:');
  problems.forEach((problem) => console.error(`- ${problem}`));
  process.exit(1);
}

console.log(`knowledge lint passed for ${managedFiles.length} managed file(s)`);
