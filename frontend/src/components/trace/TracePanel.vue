<script setup lang="ts">
import { computed, ref } from 'vue'
import { NButton, NTag } from 'naive-ui'
import { useTraceStore } from '@/stores'
import type { TraceEvent } from '@/types'

const props = defineProps<{
  runId: string
  defaultCollapsed?: boolean
  cutOnInterrupt?: boolean
}>()

const traceStore = useTraceStore()
const collapsed = ref(props.defaultCollapsed ?? false)
const viewMode = ref<'timeline' | 'card'>('timeline')

const tracesSorted = computed(() => {
  const ordered = [...traceStore.traces]
    .filter((trace) => trace.run_id === props.runId)
    .sort(
      (a, b) =>
        new Date(a.event_time).getTime() - new Date(b.event_time).getTime()
    )

  if (props.cutOnInterrupt === false) {
    return ordered
  }

  const interruptIndex = ordered.findIndex(
    (trace) => trace.trace_kind === 'hitl_interrupt'
  )
  if (interruptIndex >= 0) {
    return ordered.slice(0, interruptIndex + 1)
  }
  return ordered
})

const toggleCollapse = () => {
  collapsed.value = !collapsed.value
}

const toggleView = () => {
  viewMode.value = viewMode.value === 'timeline' ? 'card' : 'timeline'
}

const isErrorTrace = (trace: TraceEvent) =>
  trace.phase === 'error' || trace.ok === false || Boolean(trace.error)

const getTodoItems = (trace: TraceEvent) => {
  const payload = trace.payload as { todos?: Array<{ status?: string; content?: string }> }
  return Array.isArray(payload?.todos) ? payload.todos : []
}

const formatTodoStatus = (status?: string) => {
  if (status === 'completed') return '已完成'
  if (status === 'in_progress') return '进行中'
  return '待处理'
}

const todoStatusClass = (status?: string) => {
  if (status === 'completed') return 'border-green-500 text-green-600 bg-green-50/60'
  if (status === 'in_progress') return 'border-blue-500 text-blue-600 bg-blue-50/60'
  return 'border-gray-400 text-gray-600 bg-gray-50/60'
}

const formatTime = (value: string) => {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleTimeString('zh-CN', { hour12: false })
}

const buildSummary = (trace: TraceEvent) => {
  if (trace.trace_kind === 'todo_update') {
    const todos = (trace.payload as { todos?: unknown[] })?.todos
    const count = Array.isArray(todos) ? todos.length : 0
    return `TODO 更新 (${count} 项)`
  }
  if (trace.trace_kind === 'subagent_dispatch') {
    const summary = (trace.payload as { task_summary?: string })?.task_summary
    return summary ? `子代理任务：${summary}` : '子代理调度'
  }
  if (trace.trace_kind === 'hitl_interrupt') {
    const pending = (trace.payload as { pending_actions?: number })?.pending_actions
    return `审批中断 (${pending ?? 0} 个动作)`
  }
  if (trace.trace_kind === 'tool_span') {
    return trace.tool_name ? `工具调用：${trace.tool_name}` : '工具调用'
  }
  return trace.trace_kind
}
</script>

<template>
  <div class="border-t border-base bg-panel">
    <div class="flex items-center justify-between px-4 py-2">
      <div class="flex items-center gap-2">
        <span class="font-medium">思考过程</span>
        <NTag size="small" type="info">{{ tracesSorted.length }}</NTag>
      </div>
      <div class="flex items-center gap-2">
        <NButton size="tiny" quaternary @click="toggleView">
          {{ viewMode === 'timeline' ? '卡片视图' : '时间轴' }}
        </NButton>
        <NButton size="tiny" quaternary @click="toggleCollapse">
          {{ collapsed ? '展开' : '折叠' }}
        </NButton>
      </div>
    </div>

    <div v-if="!collapsed" class="max-h-56 overflow-y-auto px-4 pb-3">
      <div v-if="tracesSorted.length === 0" class="text-sm text-muted py-2">
        暂无思考过程
      </div>

      <div v-else class="flex flex-col gap-2">
        <div
          v-for="trace in tracesSorted"
          :key="trace.id"
          class="rounded border px-3 py-2 text-sm"
          :class="[
            viewMode === 'timeline' ? 'relative pl-6 border-l-2' : '',
            isErrorTrace(trace) ? 'border-red-400 bg-red-50/50' : 'border-base bg-base'
          ]"
        >
          <span
            v-if="viewMode === 'timeline'"
            class="absolute left-2 top-4 h-2 w-2 rounded-full"
            :class="isErrorTrace(trace) ? 'bg-red-500' : 'bg-accent'"
          />

          <div class="flex items-center justify-between gap-2">
            <div class="flex items-center gap-2">
              <span class="font-medium">{{ buildSummary(trace) }}</span>
              <NTag size="tiny" type="success" v-if="trace.phase === 'end'">完成</NTag>
              <NTag size="tiny" type="warning" v-else-if="trace.phase === 'start'">开始</NTag>
              <NTag size="tiny" type="error" v-else-if="trace.phase === 'error'">错误</NTag>
              <NTag size="tiny" type="info" v-else>{{ trace.phase }}</NTag>
            </div>
            <span class="text-xs text-muted">{{ formatTime(trace.event_time) }}</span>
          </div>

          <div class="mt-1 text-xs text-muted">
            <span v-if="trace.subagent_type">子代理：{{ trace.subagent_type }}</span>
            <span v-else-if="trace.tool_name">工具：{{ trace.tool_name }}</span>
            <span v-else>组件：{{ trace.component }}</span>
          </div>

          <div v-if="trace.error" class="mt-1 text-xs text-red-600">
            {{ trace.error }}
          </div>

          <div v-if="trace.trace_kind === 'todo_update' && getTodoItems(trace).length" class="mt-2 space-y-1 text-xs">
            <div
              v-for="(todo, idx) in getTodoItems(trace)"
              :key="idx"
              class="flex items-start gap-2"
            >
              <span
                class="inline-flex items-center rounded-full border px-2 py-0.5 text-[10px]"
                :class="todoStatusClass(todo.status)"
              >
                {{ formatTodoStatus(todo.status) }}
              </span>
              <span class="text-sm text-base">{{ todo.content || '（无内容）' }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
