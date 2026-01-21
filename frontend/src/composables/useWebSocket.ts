import { ref, onUnmounted } from 'vue'
import { createWebSocket, sendMessage } from '@/api/websocket'
import type { WsEvent } from '@/types'

const MAX_RECONNECT_ATTEMPTS = 5
const RECONNECT_DELAY = 3000

export function useWebSocket() {
  const ws = ref<WebSocket | null>(null)
  const connected = ref(false)
  const streamingContent = ref('')
  const isStreaming = ref(false)
  
  let reconnectAttempts = 0
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let currentUserId = ''
  let currentThreadId = ''
  let currentOnEvent: ((event: WsEvent) => void) | null = null

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
        if (event.type === 'token') {
          streamingContent.value += event.delta
          isStreaming.value = true
        } else if (event.type === 'final' || event.type === 'error' || event.type === 'approval_required') {
          isStreaming.value = false
          if (event.type === 'approval_required') {
            streamingContent.value = ''
          }
        }
        currentOnEvent?.(event)
      },
      () => {
        connected.value = true
        reconnectAttempts = 0
      },
      () => {
        connected.value = false
        isStreaming.value = false
        tryReconnect()
      },
      () => {
        connected.value = false
        isStreaming.value = false
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
    streamingContent.value = ''
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
    streamingContent.value = ''
    currentThreadId = ''
  }

  const clearStreaming = () => {
    streamingContent.value = ''
  }

  onUnmounted(disconnect)

  return {
    connected,
    streamingContent,
    isStreaming,
    connect,
    send,
    disconnect,
    clearStreaming,
  }
}
