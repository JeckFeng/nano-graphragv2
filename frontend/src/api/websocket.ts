import type { TraceWsEvent, WsEvent } from '@/types'

export type WsMessageHandler = (event: WsEvent | TraceWsEvent) => void

const normalizeWsBase = (apiBase?: string): string => {
  const base = apiBase?.trim()
  if (base) {
    if (base.startsWith('http://')) {
      return base.replace('http://', 'ws://').replace(/\/+$/, '')
    }
    if (base.startsWith('https://')) {
      return base.replace('https://', 'wss://').replace(/\/+$/, '')
    }
  }

  if (base?.startsWith('/')) {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    return `${protocol}://${window.location.host}${base.replace(/\/+$/, '')}`
  }

  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${protocol}://${window.location.host}`
}

export function createWebSocket(
  userId: string,
  threadId: string,
  onMessage: WsMessageHandler,
  onOpen?: () => void,
  onClose?: () => void,
  onError?: (error: Event) => void
): WebSocket {
  const baseUrl = import.meta.env.VITE_WS_BASE_URL || normalizeWsBase(import.meta.env.VITE_API_BASE_URL)
  const url = `${baseUrl}/v1/ws/chat?user_id=${userId}&thread_id=${threadId}&streaming=1`
  const ws = new WebSocket(url)

  ws.onopen = () => onOpen?.()
  ws.onclose = () => onClose?.()
  ws.onerror = (e) => onError?.(e)
  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data) as WsEvent | TraceWsEvent
      onMessage(data)
    } catch {
      console.error('Failed to parse WebSocket message:', e.data)
    }
  }

  return ws
}

export function sendMessage(ws: WebSocket | null, content: string): boolean {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'user_message', content }))
    return true
  }
  return false
}
