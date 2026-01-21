<script setup lang="ts">
import { NSpin } from 'naive-ui'
import { useConversationStore } from '@/stores/conversation'
import ConversationItem from './ConversationItem.vue'

const conversationStore = useConversationStore()

const emit = defineEmits<{
  select: [threadId: string]
}>()

const handleSelect = (threadId: string) => {
  emit('select', threadId)
}
</script>

<template>
  <div class="h-full overflow-y-auto">
    <NSpin :show="conversationStore.loading">
      <div v-if="conversationStore.list.length === 0" class="text-muted text-sm text-center py-8">
        暂无会话
      </div>
      <div v-else class="space-y-1">
        <ConversationItem
          v-for="conv in conversationStore.list"
          :key="conv.thread_id"
          :conversation="conv"
          :active="conv.thread_id === conversationStore.currentId"
          @select="handleSelect"
        />
      </div>
    </NSpin>
  </div>
</template>
