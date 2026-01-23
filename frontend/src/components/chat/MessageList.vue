<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { NSpin } from 'naive-ui'
import type { Message } from '@/types'
import MessageBubble from './MessageBubble.vue'
import TracePanel from '@/components/trace/TracePanel.vue'

const props = defineProps<{
  messages: Message[]
  streamingContent: string
  isStreaming: boolean
  streamingRunId?: string | null
  loading: boolean
}>()

const traceEnabled = String(import.meta.env.VITE_TRACE_PANEL_ENABLED).toLowerCase() === 'true'

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
        <div
          v-for="msg in messages"
          :key="msg.id"
          class="mb-4"
        >
          <TracePanel
            v-if="traceEnabled && msg.role === 'assistant' && msg.run_id"
            :run-id="msg.run_id"
            :default-collapsed="true"
            :cut-on-interrupt="Boolean(msg.tool_payload?.partial)"
          />
          <MessageBubble
            :message="msg"
          />
        </div>
        
        <!-- 流式输出中的消息 -->
        <div v-if="isStreaming && streamingContent" class="mb-4">
          <TracePanel
            v-if="traceEnabled && streamingRunId"
            :run-id="streamingRunId"
            :default-collapsed="false"
            :cut-on-interrupt="false"
          />
          <MessageBubble
            :message="{
              id: -1,
              role: 'assistant',
              content: streamingContent,
              run_id: streamingRunId,
              created_at: new Date().toISOString()
            }"
            :is-streaming="true"
          />
        </div>
      </template>
    </NSpin>
  </div>
</template>
