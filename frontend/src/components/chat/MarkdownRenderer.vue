<script setup lang="ts">
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import 'highlight.js/styles/github.css'

const props = defineProps<{ content: string }>()

const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  highlight: (str, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return hljs.highlight(str, { language: lang }).value
      } catch {
        // ignore
      }
    }
    return ''
  },
})

const rendered = computed(() => md.render(props.content || ''))
</script>

<template>
  <div class="markdown-body prose prose-sm max-w-none" v-html="rendered" />
</template>

<style>
.markdown-body pre {
  background-color: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
  overflow-x: auto;
}

.markdown-body code {
  background-color: var(--panel);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.875em;
}

.markdown-body pre code {
  background: none;
  padding: 0;
}

.markdown-body table {
  border-collapse: collapse;
  width: 100%;
}

.markdown-body th,
.markdown-body td {
  border: 1px solid var(--border);
  padding: 8px 12px;
}

.markdown-body th {
  background-color: var(--panel);
}
</style>
