export interface ConversationSummary {
  thread_id: string
  title: string | null
  created_at: string
}

export interface Message {
  id: number
  role: 'user' | 'assistant' | 'tool'
  content: string
  created_at: string
  artifacts?: RouteArtifact[]
}

export interface RouteArtifact {
  type: 'route_polyline'
  polylines: string[]
  origin: string
  destination: string
}

export interface ApprovalInterrupt {
  interrupt_id: string
  action_requests: Array<{ name: string; args: Record<string, unknown> }>
  review_configs: Array<{ action_name: string; allowed_decisions: string[] }>
}

export interface ApprovalInfo {
  approval_id: string
  thread_id: string
  status: 'pending' | 'approved' | 'rejected' | 'processing'
  tool_name: string
  tool_args: Record<string, unknown>
  interrupts: ApprovalInterrupt[]
}

export interface ApprovalRecord {
  approval_id: string
  thread_id: string
  user_id: string
  status: 'pending' | 'approved' | 'rejected' | 'edited' | 'processing'
  created_at: string
  resolved_at: string | null
  interrupts: ApprovalInterrupt[]
  decision: string | null
  result_content: string | null
}

export interface ApprovalResolutionResponse {
  approval_id: string
  status: 'pending' | 'approved' | 'rejected' | 'edited' | 'processing'
  result_content: string | null
  next_approval: ApprovalRecord | null
  isAsync?: boolean
}

export type WsEventType = 'token' | 'final' | 'error' | 'approval_required'

export interface WsEventBase {
  type: WsEventType
  thread_id: string
  message_id?: number
  sequence?: number
  timestamp?: string
}

export interface WsTokenEvent extends WsEventBase {
  type: 'token'
  delta: string
}

export interface WsFinalEvent extends WsEventBase {
  type: 'final'
  content: string
  artifacts?: RouteArtifact[]
}

export interface WsErrorEvent extends WsEventBase {
  type: 'error'
  code: string
  message: string
}

export interface WsApprovalRequiredEvent extends WsEventBase {
  type: 'approval_required'
  approval_id: string
  status: 'pending'
  interrupts: ApprovalInterrupt[]
}

export type WsEvent = WsTokenEvent | WsFinalEvent | WsErrorEvent | WsApprovalRequiredEvent

export type TraceKind =
  | 'todo_update'
  | 'tool_span'
  | 'subagent_dispatch'
  | 'hitl_interrupt'
  | 'hitl_resume'
  | string

export type TracePhase =
  | 'start'
  | 'end'
  | 'update'
  | 'error'
  | 'pending'
  | 'resume'
  | string

export interface TraceEvent {
  id: number
  event_time: string
  request_id: string | null
  thread_id: string
  run_id: string | null
  message_id: number | null
  user_id: string | null
  event_type: string
  event_name: string
  trace_kind: TraceKind
  phase: TracePhase
  source: string
  component: string
  tool_name: string | null
  subagent_type: string | null
  ok: boolean | null
  latency_ms: number | null
  error: string | null
  payload: Record<string, unknown>
  seq: number | null
}

export interface TraceListResponse {
  thread_id: string | null
  run_id: string | null
  limit: number
  offset: number
  traces: TraceEvent[]
}

export interface TraceWsEvent {
  event_type: 'trace'
  event_name?: string
  trace_kind: TraceKind
  phase: TracePhase
  thread_id: string
  trace_event_id?: number
  request_id?: string | null
  run_id?: string | null
  message_id?: number | null
  user_id?: string | null
  source?: string
  component?: string
  tool_name?: string | null
  subagent_type?: string | null
  ok?: boolean | null
  latency_ms?: number | null
  error?: string | null
  payload?: Record<string, unknown>
  seq?: number | null
  ts?: string
}
