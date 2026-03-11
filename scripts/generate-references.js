const { writeText, loadText } = require('./knowledge-lib.js');

const graph = JSON.parse(loadText('generated/graph.json'));
const externalNodes = graph.nodes.filter((node) => node.type === 'external-reference');
const externalById = new Map(externalNodes.map((node) => [node.id, node]));
const articleById = new Map(graph.nodes.filter((node) => node.file).map((node) => [node.id, node]));

const domainMap = new Map();
graph.edges
  .filter((edge) => edge.type === 'references' && externalById.has(edge.target))
  .forEach((edge) => {
    const source = articleById.get(edge.source);
    const external = externalById.get(edge.target);
    if (!source || !external) return;
    const domain = external.metadata.domain;
    if (!domainMap.has(domain)) domainMap.set(domain, []);
    domainMap.get(domain).push({
      title: source.metadata.title,
      link: `/${source.file.replace(/\.md$/, '')}`,
      url: external.metadata.url
    });
  });

const extendsEdges = graph.edges
  .filter((edge) => edge.type === 'extends')
  .map((edge) => ({
    source: articleById.get(edge.source),
    target: articleById.get(edge.target)
  }))
  .filter((item) => item.source && item.target);

const lines = [
  '# Generated References',
  '',
  '> Auto-generated external reference index and secondary-to-core extension map.',
  '',
  '## 外部引用按域名聚合',
  ''
];

if (domainMap.size === 0) {
  lines.push('- 当前受管内容中未发现可解析的外部 Markdown 链接。', '');
} else {
  Array.from(domainMap.keys()).sort().forEach((domain) => {
    lines.push(`### ${domain}`, '');
    domainMap.get(domain).forEach((entry) => {
      lines.push(`- [${entry.title}](${entry.link}) -> ${entry.url}`);
    });
    lines.push('');
  });
}

lines.push('## 次级知识对主路径的补充');
lines.push('');
if (extendsEdges.length === 0) {
  lines.push('- 当前未建立次级知识到核心章节的补充关系。');
} else {
  extendsEdges.forEach(({ source, target }) => {
    lines.push(`- [${source.metadata.title}](/${source.file.replace(/\.md$/, '')}) => [${target.metadata.title}](/${target.file.replace(/\.md$/, '')})`);
  });
}

writeText('generated/references.md', `${lines.join('\n')}\n`);
console.log('generated/references.md created');
