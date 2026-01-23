import { ref, onMounted, onUnmounted } from 'vue'
import { createWebSocket, sendMessage } from '@/api/websocket'
import { useTraceStore } from '@/stores'
import type { TraceWsEvent, WsEvent } from '@/types'

const MAX_RECONNECT_ATTEMPTS = 5
const RECONNECT_DELAY = 3000

export function useWebSocket() {
  const ws = ref<WebSocket | null>(null)
  const connected = ref(false)
  const streamingContent = ref('')
  const isStreaming = ref(false)
  const waitingResponse = ref(false)
  const traceStore = useTraceStore()
  const traceEnabled = String(import.meta.env.VITE_TRACE_PANEL_ENABLED).toLowerCase() === 'true'
  
  let reconnectAttempts = 0
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let currentUserId = ''
  let currentThreadId = ''
  let currentOnEvent: ((event: WsEvent) => void) | null = null

  // 页面卸载前清理连接
  const handleBeforeUnload = () => {
    if (ws.value) {
      ws.value.close(1000, 'Page unload')
    }
  }

  const connect = (
    userId: string,
    threadId: string,
    onEvent: (event: WsEvent) => void
  ) => {
    disconnect()
    currentUserId = userId
    currentThreadId = threadId
    currentOnEvent = onEvent
    reconnectAttempts = 0
    
    doConnect()
  }

  const doConnect = () => {
    ws.value = createWebSocket(
      currentUserId,
      currentThreadId,
      (event) => {
        if (traceEnabled && 'event_type' in event && event.event_type === 'trace') {
          traceStore.appendTrace(event as TraceWsEvent)
          return
        }
        if ('type' in event && event.type === 'token') {
          streamingContent.value += (event as { delta: string }).delta
          isStreaming.value = true
          waitingResponse.value = false
        } else if ('type' in event && (event.type === 'final' || event.type === 'error' || event.type === 'approval_required')) {
          isStreaming.value = false
          waitingResponse.value = false
          // approval_required 时不再清空 streamingContent，由 ChatPanel 处理
        }
        if ('type' in event) {
          currentOnEvent?.(event as WsEvent)
        }
      },
      () => {
        connected.value = true
        reconnectAttempts = 0
      },
      () => {
        connected.value = false
        isStreaming.value = false
        waitingResponse.value = false
        tryReconnect()
      },
      () => {
        connected.value = false
        isStreaming.value = false
        waitingResponse.value = false
        tryReconnect()
      }
    )
  }

  const tryReconnect = () => {
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS || !currentThreadId) return
    
    reconnectAttempts++
    reconnectTimer = setTimeout(doConnect, RECONNECT_DELAY)
  }

  const send = (content: string) => {
    if (!ws.value || ws.value.readyState !== WebSocket.OPEN) {
      return false
    }
    streamingContent.value = ''
    waitingResponse.value = true
    return sendMessage(ws.value, content)
  }

  const disconnect = () => {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    ws.value?.close()
    ws.value = null
    connected.value = false
    isStreaming.value = false
    waitingResponse.value = false
    streamingContent.value = ''
    currentThreadId = ''
  }

  const clearStreaming = () => {
    streamingContent.value = ''
  }

  onMounted(() => {
    window.addEventListener('beforeunload', handleBeforeUnload)
  })

  onUnmounted(() => {
    window.removeEventListener('beforeunload', handleBeforeUnload)
    disconnect()
  })

  return {
    connected,
    streamingContent,
    isStreaming,
    waitingResponse,
    connect,
    send,
    disconnect,
    clearStreaming,
  }
}
