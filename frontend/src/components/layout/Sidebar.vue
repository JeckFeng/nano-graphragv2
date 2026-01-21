<script setup lang="ts">
import { NButton } from 'naive-ui'
import ConversationList from '../conversation/ConversationList.vue'

defineProps<{
  collapsed: boolean
}>()

const emit = defineEmits<{
  toggle: []
  create: []
  select: [threadId: string]
}>()
</script>

<template>
  <aside
    class="h-full bg-panel border-r border-base flex flex-col transition-all duration-300"
    :class="collapsed ? 'w-0 overflow-hidden' : 'w-64'"
  >
    <!-- 侧边栏头部 -->
    <div class="h-14 flex items-center justify-between px-4 border-b border-base">
      <span class="font-semibold text-base truncate">会话列表</span>
      <NButton quaternary circle size="small" @click="emit('toggle')">
        <template #icon>
          <span class="text-lg">«</span>
        </template>
      </NButton>
    </div>
    
    <!-- 会话列表区域 -->
    <div class="flex-1 overflow-y-auto p-2">
      <ConversationList @select="(id) => emit('select', id)" />
    </div>
    
    <!-- 新建会话按钮 -->
    <div class="p-3 border-t border-base">
      <NButton block type="primary" @click="emit('create')">
        + 新建会话
      </NButton>
    </div>
  </aside>
</template>
