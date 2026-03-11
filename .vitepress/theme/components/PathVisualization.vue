<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

type GraphNode = {
  id: string
  file?: string
  metadata?: {
    title: string
    learning_stage: string
    optimization_axes: string[]
    source_scope: string
    display_order: number
  }
}

const graph = ref<{ nodes: GraphNode[] } | null>(null)

const stageLabels: Record<string, string> = {
  orientation: 'Orientation',
  foundations: 'Foundations',
  'core-techniques': 'Core Techniques',
  production: 'Production',
  advanced: 'Advanced'
}

const stageOrder = ['orientation', 'foundations', 'core-techniques', 'production', 'advanced']

const stages = computed(() => {
  const nodes = (graph.value?.nodes || [])
    .filter((node) => node.file && node.metadata?.source_scope === 'core')
    .sort((left, right) => (left.metadata?.display_order || 0) - (right.metadata?.display_order || 0))

  return stageOrder
    .map((stage) => ({
      key: stage,
      label: stageLabels[stage] || stage,
      chapters: nodes.filter((node) => node.metadata?.learning_stage === stage)
    }))
    .filter((stage) => stage.chapters.length > 0)
})

onMounted(async () => {
  graph.value = await fetch('/generated/graph.json').then((response) => response.json())
})
</script>

<template>
  <div class="path-grid">
    <article v-for="stage in stages" :key="stage.key" class="path-card">
      <p class="eyebrow">{{ stage.label }}</p>
      <ul>
        <li v-for="chapter in stage.chapters" :key="chapter.id">
          <strong>{{ chapter.metadata?.title }}</strong>
          <span>{{ chapter.metadata?.optimization_axes.join(' / ') }}</span>
        </li>
      </ul>
    </article>
  </div>
</template>

<style scoped>
.path-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  margin: 1.5rem 0;
}

.path-card {
  border: 1px solid var(--vp-c-divider);
  border-radius: 18px;
  padding: 1rem 1.1rem;
  background: linear-gradient(160deg, rgba(10, 132, 255, 0.08), rgba(16, 185, 129, 0.06));
}

.eyebrow {
  margin: 0 0 0.8rem;
  font-size: 0.8rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--vp-c-brand-1);
}

ul {
  margin: 0;
  padding: 0;
  list-style: none;
}

li + li {
  margin-top: 0.8rem;
  padding-top: 0.8rem;
  border-top: 1px solid rgba(128, 128, 128, 0.2);
}

span {
  display: block;
  margin-top: 0.2rem;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}
</style>
