import { http } from './index'
import type { TraceListResponse } from '@/types'

export interface TraceQueryParams {
  limit?: number
  offset?: number
  run_id?: string
  message_id?: number
  trace_kind?: string
  phase?: string
  order?: 'asc' | 'desc'
}

export const traceApi = {
  listByThread: (threadId: string, userId: string, params: TraceQueryParams = {}) =>
    http.get<TraceListResponse>(`/v1/conversations/${threadId}/traces`, {
      params: { user_id: userId, ...params },
    }),

  listByRun: (runId: string, userId: string, params: TraceQueryParams = {}) =>
    http.get<TraceListResponse>(`/v1/runs/${runId}/traces`, {
      params: { user_id: userId, ...params },
    }),
}
