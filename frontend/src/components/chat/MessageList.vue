<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { NSpin } from 'naive-ui'
import type { Message } from '@/types'
import MessageBubble from './MessageBubble.vue'

const props = defineProps<{
  messages: Message[]
  streamingContent: string
  isStreaming: boolean
  loading: boolean
}>()

const containerRef = ref<HTMLElement | null>(null)

const scrollToBottom = () => {
  nextTick(() => {
    if (containerRef.value) {
      containerRef.value.scrollTop = containerRef.value.scrollHeight
    }
  })
}

// 消息变化时滚动到底部
watch(() => props.messages.length, scrollToBottom)
watch(() => props.streamingContent, scrollToBottom)
</script>

<template>
  <div ref="containerRef" class="flex-1 overflow-y-auto p-4">
    <NSpin :show="loading">
      <div v-if="messages.length === 0 && !isStreaming" class="text-center text-muted py-8">
        开始对话吧
      </div>
      
      <template v-else>
        <MessageBubble
          v-for="msg in messages"
          :key="msg.id"
          :message="msg"
        />
        
        <!-- 流式输出中的消息 -->
        <MessageBubble
          v-if="isStreaming && streamingContent"
          :message="{
            id: -1,
            role: 'assistant',
            content: streamingContent,
            created_at: new Date().toISOString()
          }"
          :is-streaming="true"
        />
      </template>
    </NSpin>
  </div>
</template>
