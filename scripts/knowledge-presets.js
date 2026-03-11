const SECONDARY_FILES = [
  'docs/book-metadata.md',
  'docs/content-summary.md',
  'docs/faq.md',
  'docs/refs.md',
  'docs/keywords.md',
  'docs/topic-background.md',
  'docs/market-analysis.md',
  'docs/market-comparison.md',
  'docs/success-stories.md',
  'docs/contributors.md',
  'docs/cases/boaz-barak-ai-economics.md',
  'docs/cases/qingke-ai-infra-2025-analysis.md',
  'docs/cases/sglang-int4-qat-rl-analysis.md',
  'docs/cases/lmsys-qat-mxfp4-analysis.md',
  'docs/cases/dflash-block-diffusion-analysis.md'
];

const PRESET_METADATA = {
  'README.md': {
    type: 'article',
    topics: ['llm-inference', 'inference-economics'],
    concepts: ['cost-optimization'],
    tools: [],
    architecture_layer: ['motivation-and-economics'],
    learning_stage: 'orientation',
    optimization_axes: ['cost', 'latency'],
    related: ['chapters-chapter01-introduction', 'chapters-chapter02-technology-landscape']
  },
  'chapters/chapter01-introduction.md': {
    type: 'article',
    topics: ['llm-inference', 'inference-economics'],
    concepts: ['cost-optimization', 'latency-budget'],
    tools: [],
    architecture_layer: ['motivation-and-economics'],
    learning_stage: 'orientation',
    optimization_axes: ['cost', 'latency'],
    related: ['readme', 'chapters-chapter02-technology-landscape', 'appendix-c-benchmarks-roi']
  },
  'chapters/chapter02-technology-landscape.md': {
    type: 'article',
    topics: ['llm-inference', 'technology-trends'],
    concepts: ['cost-optimization'],
    tools: [],
    architecture_layer: ['motivation-and-economics'],
    learning_stage: 'orientation',
    optimization_axes: ['cost', 'operability'],
    related: ['chapters-chapter01-introduction', 'chapters-chapter10-production-deployment', 'chapters-chapter11-advanced-topics']
  },
  'chapters/chapter03-gpu-basics.md': {
    type: 'article',
    topics: ['gpu-architecture'],
    concepts: ['memory-bandwidth', 'throughput-engineering'],
    tools: [],
    architecture_layer: ['hardware-and-runtime'],
    learning_stage: 'foundations',
    optimization_axes: ['throughput', 'memory', 'latency'],
    related: ['chapters-chapter04-environment-setup', 'chapters-chapter05-llm-inference-basics']
  },
  'chapters/chapter04-environment-setup.md': {
    type: 'article',
    topics: ['environment-setup'],
    concepts: [],
    tools: ['docker', 'cuda', 'vllm'],
    architecture_layer: ['hardware-and-runtime'],
    learning_stage: 'foundations',
    optimization_axes: ['operability', 'latency'],
    related: ['chapters-chapter03-gpu-basics', 'chapters-chapter05-llm-inference-basics', 'appendix-b-troubleshooting']
  },
  'chapters/chapter05-llm-inference-basics.md': {
    type: 'article',
    topics: ['inference-mechanics'],
    concepts: ['kv-cache', 'paged-attention', 'continuous-batching'],
    tools: ['vllm'],
    architecture_layer: ['inference-mechanics'],
    learning_stage: 'foundations',
    optimization_axes: ['latency', 'throughput', 'memory'],
    related: ['chapters-chapter06-kv-cache-optimization', 'chapters-chapter07-request-scheduling']
  },
  'chapters/chapter06-kv-cache-optimization.md': {
    type: 'article',
    topics: ['kv-cache'],
    concepts: ['kv-cache', 'paged-attention', 'prefix-caching'],
    tools: ['vllm'],
    architecture_layer: ['optimization-techniques'],
    learning_stage: 'core-techniques',
    optimization_axes: ['memory', 'latency', 'throughput', 'cost'],
    related: ['chapters-chapter05-llm-inference-basics', 'chapters-chapter07-request-scheduling', 'chapters-chapter08-quantization']
  },
  'chapters/chapter07-request-scheduling.md': {
    type: 'article',
    topics: ['request-scheduling'],
    concepts: ['continuous-batching', 'prefill-decode-disaggregation', 'throughput-engineering'],
    tools: ['vllm', 'sglang'],
    architecture_layer: ['optimization-techniques'],
    learning_stage: 'core-techniques',
    optimization_axes: ['throughput', 'latency', 'operability'],
    related: ['chapters-chapter05-llm-inference-basics', 'chapters-chapter06-kv-cache-optimization', 'chapters-chapter10-production-deployment']
  },
  'chapters/chapter08-quantization.md': {
    type: 'article',
    topics: ['quantization'],
    concepts: ['quantization', 'int4-qat', 'precision-alignment'],
    tools: ['vllm', 'sglang', 'modelopt'],
    architecture_layer: ['optimization-techniques'],
    learning_stage: 'core-techniques',
    optimization_axes: ['memory', 'cost', 'quality', 'latency'],
    related: ['chapters-chapter06-kv-cache-optimization', 'chapters-chapter09-speculative-sampling', 'chapters-chapter11-advanced-topics']
  },
  'chapters/chapter09-speculative-sampling.md': {
    type: 'article',
    topics: ['speculative-decoding'],
    concepts: ['speculative-decoding', 'latency-budget'],
    tools: ['vllm', 'sglang'],
    architecture_layer: ['optimization-techniques'],
    learning_stage: 'core-techniques',
    optimization_axes: ['latency', 'throughput', 'quality'],
    related: [
      'chapters-chapter07-request-scheduling',
      'chapters-chapter08-quantization',
      'chapters-chapter11-advanced-topics',
      'docs-cases-dflash-block-diffusion-analysis'
    ]
  },
  'chapters/chapter10-production-deployment.md': {
    type: 'article',
    topics: ['production-deployment'],
    concepts: ['observability', 'roi-monitoring', 'cost-optimization'],
    tools: ['kubernetes', 'prometheus', 'grafana'],
    architecture_layer: ['production-systems'],
    learning_stage: 'production',
    optimization_axes: ['operability', 'cost', 'latency', 'throughput'],
    related: ['chapters-chapter07-request-scheduling', 'chapters-chapter11-advanced-topics', 'appendix-b-troubleshooting', 'appendix-c-benchmarks-roi']
  },
  'chapters/chapter11-advanced-topics.md': {
    type: 'article',
    topics: ['advanced-systems'],
    concepts: ['agent-infrastructure', 'heterogeneous-deployment', 'moe-inference', 'multimodal-inference'],
    tools: ['vllm', 'triton', 'torch-compile'],
    architecture_layer: ['frontier-and-ecosystem'],
    learning_stage: 'advanced',
    optimization_axes: ['operability', 'quality', 'latency', 'throughput'],
    related: ['chapters-chapter10-production-deployment', 'chapters-chapter08-quantization', 'chapters-chapter09-speculative-sampling']
  },
  'appendix-a-tools-resources.md': {
    type: 'reference',
    topics: ['reference-materials'],
    concepts: [],
    tools: ['huggingface', 'vllm', 'sglang', 'docker'],
    architecture_layer: ['frontier-and-ecosystem'],
    learning_stage: 'advanced',
    optimization_axes: ['operability', 'cost', 'memory'],
    related: ['chapters-chapter04-environment-setup', 'chapters-chapter08-quantization', 'docs-refs']
  },
  'appendix-b-troubleshooting.md': {
    type: 'reference',
    topics: ['troubleshooting'],
    concepts: ['observability', 'latency-budget'],
    tools: ['vllm', 'nsight-systems'],
    architecture_layer: ['production-systems'],
    learning_stage: 'production',
    optimization_axes: ['operability', 'latency', 'memory', 'throughput'],
    related: ['chapters-chapter04-environment-setup', 'chapters-chapter07-request-scheduling', 'chapters-chapter10-production-deployment']
  },
  'appendix-c-benchmarks-roi.md': {
    type: 'reference',
    topics: ['benchmarks-and-roi'],
    concepts: ['roi-monitoring', 'cost-optimization', 'throughput-engineering'],
    tools: [],
    architecture_layer: ['production-systems'],
    learning_stage: 'production',
    optimization_axes: ['cost', 'latency', 'throughput', 'quality'],
    related: ['chapters-chapter01-introduction', 'chapters-chapter10-production-deployment']
  },
  'docs/book-metadata.md': {
    type: 'reference',
    topics: ['reference-materials', 'market-analysis'],
    concepts: ['cost-optimization'],
    tools: [],
    architecture_layer: ['motivation-and-economics'],
    learning_stage: 'orientation',
    optimization_axes: ['cost', 'operability'],
    related: ['readme', 'chapters-chapter01-introduction', 'chapters-chapter02-technology-landscape']
  },
  'docs/content-summary.md': {
    type: 'reference',
    topics: ['reference-materials'],
    concepts: ['cost-optimization'],
    tools: [],
    architecture_layer: ['motivation-and-economics'],
    learning_stage: 'orientation',
    optimization_axes: ['cost'],
    related: ['readme', 'chapters-chapter01-introduction']
  },
  'docs/faq.md': {
    type: 'reference',
    topics: ['community', 'reference-materials'],
    concepts: [],
    tools: ['vllm', 'docker'],
    architecture_layer: ['production-systems'],
    learning_stage: 'orientation',
    optimization_axes: ['operability'],
    related: ['readme', 'chapters-chapter04-environment-setup', 'appendix-b-troubleshooting']
  },
  'docs/refs.md': {
    type: 'reference',
    topics: ['reference-materials'],
    concepts: [],
    tools: [],
    architecture_layer: ['frontier-and-ecosystem'],
    learning_stage: 'advanced',
    optimization_axes: ['operability', 'quality'],
    related: ['appendix-a-tools-resources', 'chapters-chapter10-production-deployment']
  },
  'docs/keywords.md': {
    type: 'reference',
    topics: ['reference-materials'],
    concepts: ['kv-cache', 'continuous-batching', 'quantization', 'speculative-decoding'],
    tools: [],
    architecture_layer: ['inference-mechanics'],
    learning_stage: 'foundations',
    optimization_axes: ['quality'],
    related: ['chapters-chapter05-llm-inference-basics', 'chapters-chapter06-kv-cache-optimization', 'chapters-chapter07-request-scheduling', 'chapters-chapter08-quantization', 'chapters-chapter09-speculative-sampling']
  },
  'docs/topic-background.md': {
    type: 'reference',
    topics: ['market-analysis'],
    concepts: ['cost-optimization'],
    tools: [],
    architecture_layer: ['motivation-and-economics'],
    learning_stage: 'orientation',
    optimization_axes: ['cost'],
    related: ['readme', 'chapters-chapter01-introduction', 'chapters-chapter02-technology-landscape']
  },
  'docs/market-analysis.md': {
    type: 'case-study',
    topics: ['market-analysis'],
    concepts: ['cost-optimization', 'roi-monitoring'],
    tools: [],
    architecture_layer: ['motivation-and-economics'],
    learning_stage: 'orientation',
    optimization_axes: ['cost'],
    related: ['readme', 'chapters-chapter01-introduction', 'chapters-chapter10-production-deployment']
  },
  'docs/market-comparison.md': {
    type: 'reference',
    topics: ['market-analysis'],
    concepts: ['cost-optimization'],
    tools: [],
    architecture_layer: ['motivation-and-economics'],
    learning_stage: 'orientation',
    optimization_axes: ['cost', 'quality'],
    related: ['readme', 'chapters-chapter02-technology-landscape']
  },
  'docs/success-stories.md': {
    type: 'case-study',
    topics: ['community', 'case-studies'],
    concepts: ['roi-monitoring', 'cost-optimization'],
    tools: [],
    architecture_layer: ['production-systems'],
    learning_stage: 'production',
    optimization_axes: ['cost', 'operability', 'throughput'],
    related: ['chapters-chapter10-production-deployment', 'appendix-c-benchmarks-roi']
  },
  'docs/contributors.md': {
    type: 'reference',
    topics: ['community'],
    concepts: [],
    tools: [],
    architecture_layer: ['frontier-and-ecosystem'],
    learning_stage: 'advanced',
    optimization_axes: ['operability'],
    related: ['readme']
  },
  'docs/cases/boaz-barak-ai-economics.md': {
    type: 'case-study',
    topics: ['case-studies', 'inference-economics'],
    concepts: ['cost-optimization'],
    tools: [],
    architecture_layer: ['motivation-and-economics'],
    learning_stage: 'orientation',
    optimization_axes: ['cost'],
    related: ['chapters-chapter01-introduction', 'chapters-chapter02-technology-landscape']
  },
  'docs/cases/qingke-ai-infra-2025-analysis.md': {
    type: 'case-study',
    topics: ['case-studies', 'advanced-systems', 'production-deployment'],
    concepts: ['prefill-decode-disaggregation', 'agent-infrastructure', 'heterogeneous-deployment'],
    tools: ['vllm', 'sglang', 'kubernetes'],
    architecture_layer: ['frontier-and-ecosystem'],
    learning_stage: 'advanced',
    optimization_axes: ['throughput', 'operability', 'cost'],
    related: ['chapters-chapter07-request-scheduling', 'chapters-chapter10-production-deployment', 'chapters-chapter11-advanced-topics']
  },
  'docs/cases/sglang-int4-qat-rl-analysis.md': {
    type: 'case-study',
    topics: ['case-studies', 'quantization'],
    concepts: ['quantization', 'int4-qat', 'precision-alignment'],
    tools: ['sglang', 'modelopt'],
    architecture_layer: ['optimization-techniques'],
    learning_stage: 'core-techniques',
    optimization_axes: ['memory', 'cost', 'quality'],
    related: ['chapters-chapter08-quantization', 'chapters-chapter10-production-deployment']
  },
  'docs/cases/lmsys-qat-mxfp4-analysis.md': {
    type: 'case-study',
    topics: ['case-studies', 'quantization'],
    concepts: ['quantization', 'int4-qat', 'precision-alignment'],
    tools: ['modelopt', 'sglang'],
    architecture_layer: ['optimization-techniques'],
    learning_stage: 'core-techniques',
    optimization_axes: ['memory', 'cost', 'quality'],
    related: ['chapters-chapter08-quantization']
  },
  'docs/cases/dflash-block-diffusion-analysis.md': {
    type: 'case-study',
    topics: ['case-studies', 'speculative-decoding'],
    concepts: ['speculative-decoding', 'throughput-engineering', 'latency-budget'],
    tools: [],
    architecture_layer: ['optimization-techniques'],
    learning_stage: 'core-techniques',
    optimization_axes: ['latency', 'throughput', 'quality'],
    related: ['chapters-chapter09-speculative-sampling', 'chapters-chapter11-advanced-topics']
  }
};

module.exports = {
  SECONDARY_FILES,
  PRESET_METADATA
};
