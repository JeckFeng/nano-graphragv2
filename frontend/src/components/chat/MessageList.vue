<script setup lang="ts">
import { ref, watch, nextTick, computed } from 'vue'
import { NSpin } from 'naive-ui'
import type { Message } from '@/types'
import MessageBubble from './MessageBubble.vue'
import TracePanel from '@/components/trace/TracePanel.vue'

const props = defineProps<{
  messages: Message[]
  streamingContent: string
  isStreaming: boolean
  streamingRunId?: string | null
  traceActiveRunId?: string | null
  loading: boolean
}>()

const traceEnabled = String(import.meta.env.VITE_TRACE_PANEL_ENABLED).toLowerCase() === 'true'

const containerRef = ref<HTMLElement | null>(null)

const hasRunId = (runId: string) =>
  props.messages.some((msg) => msg.run_id === runId) ||
  (props.streamingRunId === runId && Boolean(props.streamingContent))

const showStandaloneTrace = computed(() => {
  if (!traceEnabled) return false
  if (!props.traceActiveRunId) return false
  return !hasRunId(props.traceActiveRunId)
})

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
      <div v-if="showStandaloneTrace" class="mb-4">
        <div class="mb-2 flex items-center justify-between rounded border border-base bg-panel px-3 py-2 text-sm">
          <span class="font-medium">正在思考…</span>
          <span class="rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-700">进行中</span>
        </div>
        <TracePanel
          :run-id="traceActiveRunId!"
          :default-collapsed="false"
          :cut-on-interrupt="false"
        />
      </div>

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
