<script setup lang="ts">
import type { Message } from '@/types'
import MarkdownRenderer from './MarkdownRenderer.vue'

defineProps<{
  message: Message
  isStreaming?: boolean
}>()
</script>

<template>
  <div
    class="flex mb-4"
    :class="message.role === 'user' ? 'justify-end' : 'justify-start'"
  >
    <div
      class="max-w-[80%] rounded-lg px-4 py-3"
      :class="message.role === 'user'
        ? 'bg-bubble-user text-white'
        : 'bg-bubble-ai text-base'"
    >
      <!-- 用户消息直接显示 -->
      <template v-if="message.role === 'user'">
        <p class="whitespace-pre-wrap">{{ message.content }}</p>
      </template>
      
      <!-- AI 消息使用 Markdown 渲染 -->
      <template v-else>
        <MarkdownRenderer :content="message.content" />
        <span v-if="isStreaming" class="inline-block w-2 h-4 bg-primary animate-pulse ml-1" />
      </template>
    </div>
  </div>
</template>
