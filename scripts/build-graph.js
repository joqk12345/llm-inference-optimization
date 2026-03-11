const fs = require('node:fs');
const {
  discoverMarkdownLinks,
  getCoreSummaryEntries,
  getSecondaryFileEntries,
  isCoreKnowledgeFile,
  parseArticle,
  unique,
  writeText
} = require('./knowledge-lib.js');

const entries = [
  ...getCoreSummaryEntries().map((entry) => ({ filePath: entry.filePath, order: entry.order, scope: 'core' })),
  ...getSecondaryFileEntries().map((entry) => ({ ...entry, scope: 'secondary' }))
];

const nodes = new Map();
const nodeByFile = new Map();
const edgeSet = new Set();

function addNode(node) {
  if (!nodes.has(node.id)) {
    nodes.set(node.id, node);
  }
}

function addEdge(edge) {
  const key = JSON.stringify(edge);
  if (!edgeSet.has(key)) edgeSet.add(key);
}

function addFacetNodes(articleId, type, values) {
  unique(values).forEach((value) => {
    const id = `${type}:${value}`;
    addNode({ id, label: value, type });
    addEdge({ source: articleId, target: id, type: 'related' });
  });
}

entries.forEach(({ filePath, order, scope }) => {
  const article = parseArticle(filePath, order);
  const metadata = {
    ...article.metadata,
    source_scope: scope
  };
  const node = {
    id: metadata.id,
    label: metadata.title,
    type: metadata.type,
    file: filePath,
    metadata
  };
  addNode(node);
  nodeByFile.set(filePath, node);
});

entries.forEach(({ filePath, order, scope }) => {
  const article = parseArticle(filePath, order);
  const sourceId = article.metadata.id;

  addFacetNodes(sourceId, 'topic', article.metadata.topics);
  addFacetNodes(sourceId, 'concept', article.metadata.concepts);
  addFacetNodes(sourceId, 'tool', article.metadata.tools);
  addFacetNodes(sourceId, 'optimization-axis', article.metadata.optimization_axes);

  article.metadata.related.forEach((target) => addEdge({ source: sourceId, target, type: 'related' }));
  article.metadata.references.forEach((target) => addEdge({ source: sourceId, target, type: 'references' }));

  discoverMarkdownLinks(filePath, article.content).forEach((link) => {
    if (link.kind === 'internal') {
      const targetNode = nodeByFile.get(link.target);
      if (targetNode) {
        addEdge({ source: sourceId, target: targetNode.id, type: 'references' });
        if (scope === 'secondary' && isCoreKnowledgeFile(link.target)) {
          addEdge({ source: sourceId, target: targetNode.id, type: 'extends' });
        }
      }
      return;
    }

    if (link.kind === 'external') {
      const id = `url:${link.target}`;
      const url = new URL(link.target);
      addNode({
        id,
        label: url.hostname,
        type: 'external-reference',
        metadata: { url: link.target, domain: url.hostname }
      });
      addEdge({ source: sourceId, target: id, type: 'references' });
    }
  });

  if (scope === 'secondary') {
    article.metadata.related.forEach((target) => {
      const targetNode = nodes.get(target);
      if (targetNode && targetNode.file && isCoreKnowledgeFile(targetNode.file)) {
        addEdge({ source: sourceId, target, type: 'extends' });
      }
    });
  }
});

const readingPath = getCoreSummaryEntries()
  .map((entry) => nodeByFile.get(entry.filePath))
  .filter(Boolean);

for (let index = 1; index < readingPath.length; index += 1) {
  addEdge({ source: readingPath[index - 1].id, target: readingPath[index].id, type: 'evolution' });
}

const edges = Array.from(edgeSet).map((item) => JSON.parse(item));
const graph = {
  generatedAt: new Date().toISOString(),
  nodeCount: nodes.size,
  edgeCount: edges.length,
  nodes: Array.from(nodes.values()),
  edges
};

fs.mkdirSync('generated', { recursive: true });
writeText('generated/graph.json', JSON.stringify(graph, null, 2));
console.log(`graph generated with ${graph.nodeCount} nodes and ${graph.edgeCount} edges`);
