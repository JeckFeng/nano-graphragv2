<script setup lang="ts">
import { ref } from 'vue'
import { NButton, NTooltip, useMessage } from 'naive-ui'
import type { Message } from '@/types'
import MarkdownRenderer from './MarkdownRenderer.vue'

const props = defineProps<{
  message: Message
  isStreaming?: boolean
}>()

const toast = useMessage()
const copied = ref(false)

const copyContent = async () => {
  try {
    await navigator.clipboard.writeText(props.message.content)
    copied.value = true
    toast.success('已复制')
    setTimeout(() => { copied.value = false }, 2000)
  } catch {
    toast.error('复制失败')
  }
}
</script>

<template>
  <div
    class="flex mb-4 gap-3"
    :class="message.role === 'user' ? 'flex-row-reverse' : 'flex-row'"
  >
    <!-- 头像 -->
    <div class="flex-shrink-0">
      <div
        class="w-9 h-9 rounded-full flex items-center justify-center text-white text-sm font-medium"
        :class="message.role === 'user' ? 'bg-blue-500' : 'bg-green-600'"
      >
        {{ message.role === 'user' ? '我' : 'AI' }}
      </div>
    </div>
    
    <!-- 消息内容 -->
    <div class="max-w-[75%] group">
      <div
        class="rounded-lg px-4 py-3"
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
      
      <!-- 复制按钮 -->
      <div class="mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <NTooltip trigger="hover">
          <template #trigger>
            <NButton
              size="tiny"
              quaternary
              :type="copied ? 'success' : 'default'"
              @click="copyContent"
            >
              {{ copied ? '✓ 已复制' : '复制' }}
            </NButton>
          </template>
          复制内容
        </NTooltip>
      </div>
    </div>
  </div>
</template>
