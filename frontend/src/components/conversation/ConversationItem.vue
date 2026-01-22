<script setup lang="ts">
import { NButton, NPopconfirm } from 'naive-ui'
import type { ConversationSummary } from '@/types'

const props = defineProps<{
  conversation: ConversationSummary
  active: boolean
}>()

const emit = defineEmits<{
  select: [threadId: string]
  delete: [threadId: string]
}>()

const formatTime = (dateStr: string) => {
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}

const handleDelete = (e: Event) => {
  e.stopPropagation()
  emit('delete', props.conversation.thread_id)
}
</script>

<template>
  <div
    class="p-3 rounded-lg cursor-pointer transition-colors group relative"
    :class="active ? 'bg-primary/10 text-primary' : 'hover:bg-base'"
    @click="emit('select', conversation.thread_id)"
  >
    <div class="font-medium truncate pr-6">
      {{ conversation.title || '新会话' }}
    </div>
    <div class="text-xs text-muted mt-1">
      {{ formatTime(conversation.created_at) }}
    </div>
    
    <!-- 删除按钮 -->
    <div class="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
      <NPopconfirm
        @positive-click="handleDelete"
      >
        <template #trigger>
          <NButton size="tiny" quaternary type="error" @click.stop>
            ✕
          </NButton>
        </template>
        <div class="text-sm">
          <div class="font-medium mb-1">确认删除此会话？</div>
          <div class="text-muted">删除后无法恢复</div>
        </div>
      </NPopconfirm>
    </div>
  </div>
</template>
