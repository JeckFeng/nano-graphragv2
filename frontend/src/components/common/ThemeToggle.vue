<script setup lang="ts">
import { NButton } from 'naive-ui'
import { useThemeStore } from '@/stores/theme'

const themeStore = useThemeStore()
</script>

<template>
  <NButton quaternary circle @click="themeStore.toggle">
    <template #icon>
      <span class="text-lg">{{ themeStore.isDark ? '☀️' : '🌙' }}</span>
    </template>
  </NButton>
</template>
