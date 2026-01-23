<script setup lang="ts">
import { watch } from 'vue'
import { useMessage } from 'naive-ui'
import { useWebSocket } from '@/composables/useWebSocket'
import { useUserStore, useMessageStore } from '@/stores'
import type { Message, WsEvent } from '@/types'
import MessageList from './MessageList.vue'
import ChatInput from './ChatInput.vue'
import TracePanel from '@/components/trace/TracePanel.vue'

const props = defineProps<{
  threadId: string
}>()

const emit = defineEmits<{
  'approval-required': [approvalId: string]
}>()

const userStore = useUserStore()
const messageStore = useMessageStore()
const message = useMessage()
const traceEnabled = String(import.meta.env.VITE_TRACE_PANEL_ENABLED).toLowerCase() === 'true'

const {
  connected,
  streamingContent,
  isStreaming,
  waitingResponse,
  connect,
  send,
  disconnect,
  clearStreaming,
} = useWebSocket()

// 处理 WebSocket 事件
const handleWsEvent = (event: WsEvent) => {
  if (event.type === 'final') {
    const newMessage: Message = {
      id: event.message_id || Date.now(),
      role: 'assistant',
      content: event.content,
      created_at: event.timestamp || new Date().toISOString(),
    }
    messageStore.addMessage(newMessage)
    clearStreaming()
  } else if (event.type === 'error') {
    message.error(event.message || '发生错误')
    clearStreaming()
  } else if (event.type === 'approval_required') {
    // 保存审批前的流式内容为消息（如果有）
    if (streamingContent.value) {
      const partialMessage: Message = {
        id: Date.now(),
        role: 'assistant',
        content: streamingContent.value,
        created_at: new Date().toISOString(),
      }
      messageStore.addMessage(partialMessage)
      clearStreaming()
    }
    message.warning('需要人工审核')
    emit('approval-required', event.approval_id)
  }
}

// 发送消息
const handleSend = (content: string) => {
  if (!connected.value) {
    message.error('未连接到服务器')
    return
  }

  const sent = send(content)
  if (!sent) {
    message.error('发送失败，请稍后重试')
    return
  }
  
  const userMessage: Message = {
    id: Date.now(),
    role: 'user',
    content,
    created_at: new Date().toISOString(),
  }
  messageStore.addMessage(userMessage)
}

// 监听 threadId 变化，重新连接
watch(
  () => props.threadId,
  (newId) => {
    if (newId) {
      connect(userStore.userId, newId, handleWsEvent)
    } else {
      disconnect()
    }
  },
  { immediate: true }
)
</script>

<template>
  <div class="h-full flex flex-col bg-base">
    <!-- 连接状态 -->
    <div v-if="!connected" class="px-4 py-2 bg-accent/20 text-accent text-sm text-center">
      正在连接...
    </div>
    
    <!-- 思考中提示 -->
    <div v-else-if="waitingResponse && !isStreaming" class="px-4 py-2 bg-blue-500/20 text-blue-600 text-sm text-center">
      智能体正在思考...
    </div>
    
    <!-- 消息列表 -->
    <MessageList
      :messages="messageStore.messages"
      :streaming-content="streamingContent"
      :is-streaming="isStreaming"
      :loading="messageStore.loading"
    />

    <TracePanel v-if="traceEnabled" :thread-id="threadId" />
    
    <!-- 输入框 -->
    <ChatInput
      :disabled="!connected || isStreaming || waitingResponse"
      @send="handleSend"
    />
  </div>
</template>
