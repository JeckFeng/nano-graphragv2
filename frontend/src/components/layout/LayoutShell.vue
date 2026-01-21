<script setup lang="ts">
import { ref } from 'vue'
import Sidebar from './Sidebar.vue'
import Header from './Header.vue'

const sidebarCollapsed = ref(false)

const emit = defineEmits<{
  create: []
  select: [threadId: string]
}>()

const toggleSidebar = () => {
  sidebarCollapsed.value = !sidebarCollapsed.value
}
</script>

<template>
  <div class="flex h-screen bg-base">
    <!-- 左侧边栏 -->
    <Sidebar
      :collapsed="sidebarCollapsed"
      @toggle="toggleSidebar"
      @create="emit('create')"
      @select="(id) => emit('select', id)"
    />
    
    <!-- 右侧主区域 -->
    <div class="flex-1 flex flex-col min-w-0">
      <!-- 顶部工具栏 -->
      <Header @toggle-sidebar="toggleSidebar" />
      
      <!-- 主内容区 -->
      <main class="flex-1 overflow-hidden">
        <slot />
      </main>
    </div>
  </div>
</template>
