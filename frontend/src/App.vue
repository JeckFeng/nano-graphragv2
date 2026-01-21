<script setup lang="ts">
import { NConfigProvider, NMessageProvider, darkTheme } from 'naive-ui'
import { useThemeStore } from '@/stores'
import AppContent from '@/components/AppContent.vue'

const themeStore = useThemeStore()
</script>

<template>
  <NConfigProvider :theme="themeStore.isDark ? darkTheme : undefined">
    <NMessageProvider>
      <AppContent />
    </NMessageProvider>
  </NConfigProvider>
</template>
