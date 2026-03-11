<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

type GraphNode = {
  id: string
  file?: string
  metadata?: {
    title: string
    architecture_layer: string[]
    source_scope: string
    display_order: number
  }
}

const graph = ref<{ nodes: GraphNode[] } | null>(null)

const layers = computed(() => {
  const nodes = (graph.value?.nodes || [])
    .filter((node) => node.file && node.metadata?.source_scope === 'core')
    .sort((left, right) => (left.metadata?.display_order || 0) - (right.metadata?.display_order || 0))

  const grouped = new Map<string, GraphNode[]>()
  nodes.forEach((node) => {
    ;(node.metadata?.architecture_layer || []).forEach((layer) => {
      const bucket = grouped.get(layer) || []
      bucket.push(node)
      grouped.set(layer, bucket)
    })
  })

  return Array.from(grouped.entries())
})

onMounted(async () => {
  graph.value = await fetch('/generated/graph.json').then((response) => response.json())
})
</script>

<template>
  <div class="layer-stack">
    <section v-for="[layer, chapters] in layers" :key="layer" class="layer-row">
      <h3>{{ layer }}</h3>
      <div class="pill-wrap">
        <span v-for="chapter in chapters" :key="chapter.id" class="pill">
          {{ chapter.metadata?.title }}
        </span>
      </div>
    </section>
  </div>
</template>

<style scoped>
.layer-stack {
  display: grid;
  gap: 1rem;
  margin: 1.5rem 0;
}

.layer-row {
  border-left: 4px solid var(--vp-c-brand-1);
  padding: 0.8rem 0 0.8rem 1rem;
  background: linear-gradient(90deg, rgba(244, 114, 182, 0.08), rgba(59, 130, 246, 0.04));
  border-radius: 14px;
}

h3 {
  margin: 0 0 0.75rem;
  font-size: 1rem;
}

.pill-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
}

.pill {
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  padding: 0.35rem 0.7rem;
  background: rgba(255, 255, 255, 0.6);
  font-size: 0.9rem;
}
</style>
