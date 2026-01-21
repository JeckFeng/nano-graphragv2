<script setup lang="ts">
import type { ConversationSummary } from '@/types'

defineProps<{
  conversation: ConversationSummary
  active: boolean
}>()

const emit = defineEmits<{
  select: [threadId: string]
}>()

const formatTime = (dateStr: string) => {
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}
</script>

<template>
  <div
    class="p-3 rounded-lg cursor-pointer transition-colors"
    :class="active ? 'bg-primary/10 text-primary' : 'hover:bg-base'"
    @click="emit('select', conversation.thread_id)"
  >
    <div class="font-medium truncate">
      {{ conversation.title || '新会话' }}
    </div>
    <div class="text-xs text-muted mt-1">
      {{ formatTime(conversation.created_at) }}
    </div>
  </div>
</template>
