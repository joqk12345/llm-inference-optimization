import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import PathVisualization from './components/PathVisualization.vue'
import ArchitectureVisualization from './components/ArchitectureVisualization.vue'

const theme: Theme = {
  ...DefaultTheme,
  enhanceApp({ app }) {
    app.component('PathVisualization', PathVisualization)
    app.component('ArchitectureVisualization', ArchitectureVisualization)
  }
}

export default theme
