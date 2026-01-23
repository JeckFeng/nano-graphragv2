<script setup lang="ts">
import { NList, NListItem, NTag, NEmpty } from 'naive-ui'
import type { ApprovalRecord } from '@/types'

defineProps<{
  approvals: ApprovalRecord[]
  selectedId: string | null
}>()

const emit = defineEmits<{
  select: [id: string]
}>()

const getToolName = (approval: ApprovalRecord) => {
  return approval.interrupts[0]?.action_requests[0]?.name || '未知工具'
}

const truncateLabel = (value: string, maxLength = 16) => {
  if (value.length <= maxLength) return value
  return `${value.slice(0, maxLength)}…`
}

const formatCreatedAt = (value: string) => {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
}
</script>

<template>
  <div class="h-full overflow-y-auto">
    <NEmpty v-if="approvals.length === 0" description="暂无待审批项" class="py-8" />
    
    <NList v-else hoverable clickable>
      <NListItem
        v-for="item in approvals"
        :key="item.approval_id"
        :class="{ 'bg-accent/10': selectedId === item.approval_id }"
        @click="emit('select', item.approval_id)"
      >
        <div class="flex items-center justify-between">
          <div>
            <div class="font-medium max-w-[140px] truncate">
              {{ truncateLabel(getToolName(item)) }}
            </div>
            <div class="text-xs text-muted mt-1">{{ formatCreatedAt(item.created_at) }}</div>
          </div>
          <NTag size="small" :type="item.status === 'pending' ? 'warning' : 'success'">
            {{ item.status }}
          </NTag>
        </div>
      </NListItem>
    </NList>
  </div>
</template>
