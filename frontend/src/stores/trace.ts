import { defineStore } from 'pinia'
import { ref } from 'vue'
import { traceApi, type TraceQueryParams } from '@/api/traces'
import type { TraceEvent, TraceWsEvent } from '@/types'

export const useTraceStore = defineStore('trace', () => {
  const traces = ref<TraceEvent[]>([])
  const loading = ref(false)

  const normalizeTrace = (raw: TraceEvent | TraceWsEvent): TraceEvent => {
    const rawId = (raw as TraceEvent).id ?? (raw as TraceWsEvent).trace_event_id ?? Date.now()
    const eventTime =
      (raw as TraceEvent).event_time ?? (raw as TraceWsEvent).ts ?? new Date().toISOString()
    return {
      id: Number(rawId),
      event_time: eventTime,
      request_id: (raw as TraceEvent).request_id ?? (raw as TraceWsEvent).request_id ?? null,
      thread_id: (raw as TraceEvent).thread_id ?? (raw as TraceWsEvent).thread_id ?? '',
      run_id: (raw as TraceEvent).run_id ?? (raw as TraceWsEvent).run_id ?? null,
      message_id: (raw as TraceEvent).message_id ?? (raw as TraceWsEvent).message_id ?? null,
      user_id: (raw as TraceEvent).user_id ?? (raw as TraceWsEvent).user_id ?? null,
      event_type: (raw as TraceEvent).event_type ?? (raw as TraceWsEvent).event_type ?? 'trace',
      event_name: (raw as TraceEvent).event_name ?? (raw as TraceWsEvent).event_name ?? 'trace_event',
      trace_kind: (raw as TraceEvent).trace_kind ?? (raw as TraceWsEvent).trace_kind ?? 'tool_span',
      phase: (raw as TraceEvent).phase ?? (raw as TraceWsEvent).phase ?? 'update',
      source: (raw as TraceEvent).source ?? (raw as TraceWsEvent).source ?? 'backend',
      component: (raw as TraceEvent).component ?? (raw as TraceWsEvent).component ?? 'agent',
      tool_name: (raw as TraceEvent).tool_name ?? (raw as TraceWsEvent).tool_name ?? null,
      subagent_type: (raw as TraceEvent).subagent_type ?? (raw as TraceWsEvent).subagent_type ?? null,
      ok: (raw as TraceEvent).ok ?? (raw as TraceWsEvent).ok ?? null,
      latency_ms: (raw as TraceEvent).latency_ms ?? (raw as TraceWsEvent).latency_ms ?? null,
      error: (raw as TraceEvent).error ?? (raw as TraceWsEvent).error ?? null,
      payload: (raw as TraceEvent).payload ?? (raw as TraceWsEvent).payload ?? {},
      seq: (raw as TraceEvent).seq ?? (raw as TraceWsEvent).seq ?? null,
    }
  }

  const loadThreadTraces = async (threadId: string, userId: string, params: TraceQueryParams = {}) => {
    loading.value = true
    try {
      const res = await traceApi.listByThread(threadId, userId, params)
      traces.value = res.data.traces
    } finally {
      loading.value = false
    }
  }

  const loadRunTraces = async (runId: string, userId: string, params: TraceQueryParams = {}) => {
    loading.value = true
    try {
      const res = await traceApi.listByRun(runId, userId, params)
      traces.value = res.data.traces
    } finally {
      loading.value = false
    }
  }

  const appendTrace = (raw: TraceEvent | TraceWsEvent) => {
    const normalized = normalizeTrace(raw)
    if (!normalized.thread_id) return
    if (traces.value.some((item) => item.id === normalized.id)) {
      return
    }
    traces.value.push(normalized)
  }

  const clearTraces = () => {
    traces.value = []
  }

  return { traces, loading, loadThreadTraces, loadRunTraces, appendTrace, clearTraces }
})
