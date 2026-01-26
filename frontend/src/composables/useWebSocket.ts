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
  const streamingRunId = ref<string | null>(null)
  const traceActiveRunId = ref<string | null>(null)
  const traceStore = useTraceStore()
  const traceEnabled = String(import.meta.env.VITE_TRACE_PANEL_ENABLED).toLowerCase() === 'true'
  const traceQueueDebug = String(import.meta.env.VITE_TRACE_QUEUE_DEBUG).toLowerCase() === 'true'
  
  const traceQueue: TraceWsEvent[] = []
  let flushingTraceQueue = false
  let lastFlushAt: number | null = null
  const TRACE_QUEUE_PER_TICK = 2
  const TRACE_QUEUE_INTERVAL_MS = 1000

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
          enqueueTraceEvent(event as TraceWsEvent)
          const runId = (event as TraceWsEvent).run_id
          if (runId) {
            traceActiveRunId.value = runId
          }
          return
        }
        if ('type' in event && event.type === 'token') {
          streamingContent.value += (event as { delta: string }).delta
          if (event.run_id) {
            streamingRunId.value = event.run_id
          }
          isStreaming.value = true
          waitingResponse.value = false
        } else if ('type' in event && (event.type === 'final' || event.type === 'error' || event.type === 'approval_required')) {
          isStreaming.value = false
          waitingResponse.value = false
          if (event.type === 'final' || event.type === 'error') {
            streamingRunId.value = null
            traceActiveRunId.value = null
          }
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
    streamingRunId.value = null
    traceActiveRunId.value = null
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
    streamingRunId.value = null
    traceActiveRunId.value = null
    currentThreadId = ''
  }

  const clearStreaming = () => {
    streamingContent.value = ''
  }

  const enqueueTraceEvent = (event: TraceWsEvent) => {
    traceQueue.push(event)
    if (traceQueueDebug) {
      console.debug('[trace-queue] enqueue', {
        trace_kind: event.trace_kind,
        run_id: event.run_id,
        queue_length: traceQueue.length,
        ts: event.ts,
      })
    }
    if (!flushingTraceQueue) {
      flushingTraceQueue = true
      setTimeout(flushTraceQueue, TRACE_QUEUE_INTERVAL_MS)
    }
  }

  const flushTraceQueue = () => {
    let flushed = 0
    while (flushed < TRACE_QUEUE_PER_TICK && traceQueue.length > 0) {
      const next = traceQueue.shift()
      if (!next) break
      traceStore.appendTrace(next)
      flushed += 1
    }
    if (traceQueueDebug) {
      const now = performance.now()
      const delta = lastFlushAt ? Math.round(now - lastFlushAt) : null
      lastFlushAt = now
      console.debug('[trace-queue] flush', {
        flushed,
        remaining: traceQueue.length,
        tick_delta_ms: delta,
      })
    }
    if (traceQueue.length > 0) {
      setTimeout(flushTraceQueue, TRACE_QUEUE_INTERVAL_MS)
    } else {
      flushingTraceQueue = false
      lastFlushAt = null
    }
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
    streamingRunId,
    traceActiveRunId,
    connect,
    send,
    disconnect,
    clearStreaming,
  }
}
